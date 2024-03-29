import argparse
import json
import logging
import os
import subprocess
import sys
import re
import xml.etree.ElementTree as ET

sys.path.append('../../../')
sys.path.append('../../../comment_update')


class DiffTreeNode:
    def __init__(self, value, attribute, src, is_leaf):
        self.value = value
        self.node_id = -1
        self.parents = []
        self.attribute = attribute
        self.src = src
        self.is_leaf = is_leaf
        self.children = []
        self.prev_siblings = []
        self.next_siblings = []
        self.aligned_neighbors = []
        self.action_type = None
        self.prev_tokens = []
        self.next_tokens = []
        self.subtokens = []

        self.subtoken_children = []
        self.subtoken_parents = []
        self.prev_subtokens = []
        self.next_subtokens = []

    def to_json(self):
        return {
            'value': self.value,
            'node_id': self.node_id,
            'parent_ids': [p.node_id for p in self.parents],
            'attribute': self.attribute,
            'src': self.src,
            'is_leaf': self.is_leaf,
            'children_ids': [c.node_id for c in self.children],
            'prev_sibling_ids': [p.node_id for p in self.prev_siblings],
            'next_sibling_ids': [n.node_id for n in self.next_siblings],
            'aligned_neighbor_ids': [n.node_id for n in self.aligned_neighbors],
            'action_type': self.action_type,
        }

    @property
    def is_identifier(self):
        return self.is_leaf and self.attribute == 'SimpleName'


class DiffAST:
    def __init__(self, ast_root):
        self.node_cache = set()
        self.root = ast_root
        self.nodes = []
        self.traverse(self.root)

    def traverse(self, curr_node):
        if curr_node not in self.node_cache:
            self.node_cache.add(curr_node)
            curr_node.node_id = len(self.nodes)
            self.nodes.append(curr_node)
        for child in curr_node.subtoken_children:
            self.traverse(child)
        for child in curr_node.children:
            self.traverse(child)

    def to_json(self):
        return [n.to_json() for n in self.nodes]

    @property
    def leaves(self):
        return [n for n in self.nodes if n.is_leaf]

    @classmethod
    def from_json(cls, obj):
        nodes = []
        for node_obj in obj:
            node = DiffTreeNode(node_obj['value'], node_obj['attribute'], node_obj['src'], False)
            if 'action_type' in node_obj:
                node.action_type = node_obj['action_type']
            nodes.append(node)

        new_nodes = []

        for n, node_obj in enumerate(obj):
            nodes[n].parents = [nodes[i] for i in node_obj['parent_ids']]
            nodes[n].children = [nodes[i] for i in node_obj['children_ids']]
            nodes[n].prev_siblings = [nodes[i] for i in node_obj['prev_sibling_ids']]
            nodes[n].next_siblings = [nodes[i] for i in node_obj['next_sibling_ids']]
            nodes[n].aligned_neighbors = [nodes[i] for i in node_obj['aligned_neighbor_ids']]
            new_nodes.append(nodes[n])

            if len(nodes[n].children) == 0:
                nodes[n].is_leaf = True
                curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', nodes[n].value).split()
                new_curr = []
                for c in curr:
                    by_symbol = re.findall(r"[a-zA-Z0-9]+|[^\sa-zA-Z0-9]|[^_\sa-zA-Z0-9]", c.strip())
                    new_curr = new_curr + by_symbol
                nodes[n].subtokens = [s.lower() for s in new_curr]

                if len(nodes[n].subtokens) > 1:
                    for s in nodes[n].subtokens:
                        sub_node = DiffTreeNode(s, '', nodes[n].src, True)
                        sub_node.action_type = nodes[n].action_type
                        sub_node.subtoken_parents.append(nodes[n])

                        if len(nodes[n].subtoken_children) > 0:
                            nodes[n].subtoken_children[-1].next_subtokens.append(sub_node)
                            sub_node.prev_subtokens.append(nodes[n].subtoken_children[-1])

                        nodes[n].subtoken_children.append(sub_node)
                        new_nodes.append(sub_node)

                nodes[n].value = nodes[n].value.lower()

        return cls(new_nodes[0])


class Indexer:
    def __init__(self):
        self.count = 0

    def generate(self):
        new_id = self.count
        self.count += 1
        return new_id


class XMLNode:
    def __init__(self, value, node_id, parent, attribute,
                 alignment_id, location_id, src, is_leaf=True):
        self.value = value
        self.node_id = node_id
        self.parent = parent
        self.attribute = attribute
        self.alignment_id = alignment_id
        self.location_id = location_id
        self.src = src
        self.is_leaf = is_leaf
        self.children = []
        self.pseudo_children = []
        self.prev_sibling = None
        self.next_sibling = None

    def print_node(self):
        parent_value = None
        if self.parent:
            parent_value = self.parent.value

        print('{}: {} ({}, {})'.format(self.node_id, self.value, parent_value, len(self.children)))
        for c in self.children:
            c.print_node()


class AST:
    def __init__(self, ast_root):
        self.root = ast_root
        self.nodes = []
        self.traverse(ast_root)

    def traverse(self, curr_node):
        self.nodes.append(curr_node)
        for c, child_node in enumerate(curr_node.children):
            if c > 0:
                child_node.prev_sibling = curr_node.children[c - 1]
            if c < len(curr_node.children) - 1:
                child_node.next_sibling = curr_node.children[c + 1]
            self.traverse(child_node)

    @property
    def leaves(self):
        return [n for n in self.nodes if n.is_leaf]


def parse_xml_obj(xml_obj, indexer, parent, src):
    fields = xml_obj.attrib
    attribute = fields['typeLabel']
    is_leaf = False

    if 'label' in fields:
        is_leaf = True
        value = fields['label']
    else:
        value = attribute

    alignment_id = None
    location_id = '{}-{}-{}-{}'.format(fields['type'], value, fields['pos'], fields['length'])

    if 'other_pos' in fields:
        if src == 'old':
            alignment_id = '{}-{}-{}-{}'.format(fields['pos'], fields['length'], fields['other_pos'],
                                                fields['other_length'])
        else:
            alignment_id = '{}-{}-{}-{}'.format(fields['other_pos'], fields['other_length'], fields['pos'],
                                                fields['length'])

    node = XMLNode(value, indexer.generate(), parent,
                   attribute, alignment_id, location_id, src, is_leaf)

    for child_obj in xml_obj:
        node.children.append(parse_xml_obj(child_obj, indexer, node, src))
    return node


def set_id(diff_node, indexer):
    diff_node.node_id = indexer.generate()
    for node in diff_node.children:
        set_id(node, indexer)


def print_diff_node(diff_node):
    print('{} ({}-{}): {}, {}'.format(diff_node.value, diff_node.src, diff_node.node_id,
                                      [c.value for c in diff_node.children], [p.node_id for p in diff_node.parents]))
    for child in diff_node.children:
        print_diff_node(child)


def get_individual_ast_objs(old_sample_path, new_sample_path, actions_json, jar_path):
    old_xml_path = os.path.join(XML_DIR, 'old.xml')
    new_xml_path = os.path.join(XML_DIR, 'new.xml')

    output = subprocess.check_output(['java', '-jar', jar_path, old_sample_path,
                                      new_sample_path, old_xml_path, new_xml_path, actions_json])

    xml_obj = ET.parse(old_xml_path)
    old_root = parse_xml_obj(xml_obj.getroot()[1], Indexer(), None, 'old')
    old_ast = AST(old_root)

    xml_obj = ET.parse(new_xml_path)
    new_root = parse_xml_obj(xml_obj.getroot()[1], Indexer(), None, 'new')
    new_ast = AST(new_root)

    old_nodes = old_ast.nodes
    old_diff_nodes = [DiffTreeNode(n.value, n.attribute, n.src, n.is_leaf) for n in old_nodes]

    old_diff_nodes_by_alignment = dict()
    for n, old_node in enumerate(old_nodes):
        old_diff_node = old_diff_nodes[n]
        if old_node.parent:
            old_diff_node.parents.append(old_diff_nodes[old_node.parent.node_id])

        for c in old_node.children:
            old_diff_node.children.append(old_diff_nodes[c.node_id])

        if old_node.prev_sibling:
            old_diff_node.prev_siblings.append(old_diff_nodes[old_node.prev_sibling.node_id])

        if old_node.next_sibling:
            old_diff_node.next_siblings.append(old_diff_nodes[old_node.next_sibling.node_id])

        if old_node.alignment_id:
            old_diff_nodes_by_alignment[old_node.alignment_id] = old_diff_node

    new_nodes = new_ast.nodes
    new_diff_nodes = [DiffTreeNode(n.value, n.attribute, n.src, n.is_leaf) for n in new_nodes]

    for n, new_node in enumerate(new_nodes):
        new_diff_node = new_diff_nodes[n]
        if new_node.parent:
            new_diff_node.parents.append(new_diff_nodes[new_node.parent.node_id])

        for c in new_node.children:
            new_diff_node.children.append(new_diff_nodes[c.node_id])

        if new_node.prev_sibling:
            new_diff_node.prev_siblings.append(new_diff_nodes[new_node.prev_sibling.node_id])

        if new_node.next_sibling:
            new_diff_node.next_siblings.append(new_diff_nodes[new_node.next_sibling.node_id])

    old_diff_ast = DiffAST(old_diff_nodes[0])
    new_diff_ast = DiffAST(new_diff_nodes[0])

    return old_diff_ast, new_diff_ast


def get_diff_ast(old_sample_path, new_sample_path, actions_json, jar_path):
    old_xml_path = 'old.xml'
    new_xml_path = 'new.xml'
    output = subprocess.check_output(['java', '-jar', jar_path, old_sample_path,
                                      new_sample_path, old_xml_path, new_xml_path, actions_json])

    xml_obj = ET.parse(old_xml_path)
    old_root = parse_xml_obj(xml_obj.getroot()[1], Indexer(), None, 'old')
    old_ast = AST(old_root)

    xml_obj = ET.parse(new_xml_path)
    new_root = parse_xml_obj(xml_obj.getroot()[1], Indexer(), None, 'new')
    new_ast = AST(new_root)

    with open(actions_json) as f:
        actions = json.load(f)

    old_actions = dict()
    new_actions = dict()

    for action in actions:
        location_id = '{}-{}-{}-{}'.format(action['type'], action['label'], action['position'], action['length'])
        if action['action'] == 'Insert':
            new_actions[location_id] = action['action']
        else:
            old_actions[location_id] = action['action']

    old_nodes = old_ast.nodes
    old_diff_nodes = []
    for n in old_nodes:
        old_diff_node = DiffTreeNode(n.value, n.attribute, n.src, n.is_leaf)
        if n.location_id in old_actions:
            old_diff_node.action_type = old_actions[n.location_id]
        old_diff_nodes.append(old_diff_node)

    old_diff_nodes_by_alignment = dict()
    for n, old_node in enumerate(old_nodes):
        old_diff_node = old_diff_nodes[n]
        if old_node.parent:
            old_diff_node.parents.append(old_diff_nodes[old_node.parent.node_id])

        for c in old_node.children:
            old_diff_node.children.append(old_diff_nodes[c.node_id])

        if old_node.prev_sibling:
            old_diff_node.prev_siblings.append(old_diff_nodes[old_node.prev_sibling.node_id])

        if old_node.next_sibling:
            old_diff_node.next_siblings.append(old_diff_nodes[old_node.next_sibling.node_id])

        if old_node.alignment_id:
            if old_node.alignment_id not in old_diff_nodes_by_alignment:
                old_diff_nodes_by_alignment[old_node.alignment_id] = []
            old_diff_nodes_by_alignment[old_node.alignment_id].append(old_diff_node)

    new_nodes = new_ast.nodes
    new_diff_nodes = []

    for n, new_node in enumerate(new_nodes):
        if new_node.alignment_id in old_diff_nodes_by_alignment and len(
                old_diff_nodes_by_alignment[new_node.alignment_id]) > 0:
            old_diff_node = old_diff_nodes_by_alignment[new_node.alignment_id].pop(0)
            if new_node.value == old_diff_node.value:
                new_diff_node = old_diff_node
                new_diff_node.src = 'both'
                new_diff_nodes.append(new_diff_node)
            else:
                new_diff_node = DiffTreeNode(new_node.value, new_node.attribute, new_node.src, new_node.is_leaf)
                new_diff_node.aligned_neighbors.append(old_diff_node)
                old_diff_node.aligned_neighbors.append(new_diff_node)
                new_diff_node.action_type = old_diff_node.action_type

                if new_node.location_id in new_actions:
                    new_diff_node.action_type = new_actions[new_node.location_id]

                new_diff_nodes.append(new_diff_node)
        else:
            new_diff_node = DiffTreeNode(new_node.value, new_node.attribute, new_node.src, new_node.is_leaf)
            if new_node.location_id in new_actions:
                new_diff_node.action_type = new_actions[new_node.location_id]
            new_diff_nodes.append(new_diff_node)

    for n, new_node in enumerate(new_nodes):
        new_diff_node = new_diff_nodes[n]
        if new_node.parent and new_diff_nodes[new_node.parent.node_id] not in new_diff_node.parents:
            new_diff_node.parents.append(new_diff_nodes[new_node.parent.node_id])

        for c in new_node.children:
            if new_diff_nodes[c.node_id] not in new_diff_node.children:
                new_diff_node.children.append(new_diff_nodes[c.node_id])

        if new_node.prev_sibling and new_diff_nodes[new_node.prev_sibling.node_id] not in new_diff_node.prev_siblings:
            new_diff_node.prev_siblings.append(new_diff_nodes[new_node.prev_sibling.node_id])

        if new_node.next_sibling and new_diff_nodes[new_node.next_sibling.node_id] not in new_diff_node.next_siblings:
            new_diff_node.next_siblings.append(new_diff_nodes[new_node.next_sibling.node_id])

    super_root = DiffTreeNode('SuperRoot', 'SuperRoot', 'both', False)
    super_root.children.append(old_diff_nodes[0])
    old_diff_nodes[0].parents.append(super_root)

    if old_diff_nodes[0] != new_diff_nodes[0]:
        super_root.children.append(new_diff_nodes[0])
        new_diff_nodes[0].parents.append(super_root)

    diff_ast = DiffAST(super_root)
    return diff_ast


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--old_sample_path', help='path to java file containing old version of method')
    parser.add_argument('--new_sample_path', help='path to java file containing new version of method')
    parser.add_argument('--jar_path', help='path to downloaded jar file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')
    logging.basicConfig(level=logging.ERROR, format='%(asctime)-15s %(message)s')

    XML_DIR = 'xml_files/'
    os.makedirs(XML_DIR, exist_ok=True)

    old_ast, new_ast = get_individual_ast_objs(args.old_sample_path, args.new_sample_path, 'old_new_ast_actions.json',
                                               args.jar_path)
    diff_ast = get_diff_ast(args.old_sample_path, args.new_sample_path, 'diff_ast_actions.json', args.jar_path)

    print(diff_ast.to_json())
