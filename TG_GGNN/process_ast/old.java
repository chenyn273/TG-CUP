private static ArrayList parseDNchainPattern(String dnChain) {
			if (dnChain == null) {
				throw new IllegalArgumentException(
						"The DN chain must not be null.");
			}
			ArrayList parsed = new ArrayList();
			int startIndex = 0;
			startIndex = skipSpaces(dnChain, startIndex);
			while (startIndex < dnChain.length()) {
				int endIndex = startIndex;
				boolean inQuote = false;
				out: while (endIndex < dnChain.length()) {
					char c = dnChain.charAt(endIndex);
					switch (c) {
						case '"' :
							inQuote = !inQuote;
							break;
						case '\\' :
							endIndex++; // skip the escaped char
							break;
						case ';' :
							if (!inQuote)
								break out;
					}
					endIndex++;
				}
				if (endIndex > dnChain.length()) {
					throw new IllegalArgumentException("unterminated escape");
				}
				parsed.add(dnChain.substring(startIndex, endIndex));
				startIndex = endIndex + 1;
				startIndex = skipSpaces(dnChain, startIndex);
			}
			parseDNchain(parsed);
			return parsed;
		}