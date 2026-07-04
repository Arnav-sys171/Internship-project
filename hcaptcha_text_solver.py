import re

def solve_hcaptcha_text(prompt: str) -> str:
    """
    Given an hCaptcha text challenge prompt, attempts to solve it using heuristic rules.
    Returns the solved string, or None if no heuristic matched.
    """
    prompt = prompt.strip()
    
    def extract_index(word: str):
        word = word.lower().strip()
        words = {'starting': 1, 'initial': 1, 'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9}
        if word in words: return words[word]
        try:
            return int(re.sub(r'\D', '', word))
        except:
            return -1

    # Templates 1-3: Character replacement
    # "Only if the ending is 8, replace the last character with 9 in 245666."
    match1 = re.search(r"Only if the ending is (\w+), replace the last character with (\w+) in (\d+)", prompt, re.IGNORECASE)
    # "When the final character is 4, change only that last character to 1 in 701717."
    match2 = re.search(r"When the final character is (\w+), change only that last character to (\w+) in (\d+)", prompt, re.IGNORECASE)
    # "Replace the last character with 4 only when the ending character is 8 in 235008."
    match3 = re.search(r"Replace the last character with (\w+) only when the ending character is (\w+) in (\d+)", prompt, re.IGNORECASE)
    
    # "If it ends with 1, replace that final 1 with 8 in 714061."
    match4 = re.search(r"If it ends with (\w+), replace that final \w+ with (\w+) in (\d+)", prompt, re.IGNORECASE)

    for match, (e_idx, r_idx, t_idx) in [(match1, (1, 2, 3)), (match2, (1, 2, 3)), (match3, (2, 1, 3)), (match4, (1, 2, 3))]:
        if match:
            ending_char = match.group(e_idx)
            replace_char = match.group(r_idx)
            target_string = match.group(t_idx)
            
            if target_string.endswith(ending_char):
                return target_string[:-1] + replace_char
            else:
                return target_string

    # Templates 4-5: Number collection extraction
    match = re.search(r"(starting|initial|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|\d+(?:st|nd|rd|th))\s+(?:group|collection|set).*?:\s*(.+?)(?:Please try again|$)", prompt, re.IGNORECASE)
    
    if match:
        index_word = match.group(1)
        list_str = match.group(2)
        index = extract_index(index_word) - 1 # 1-based to 0-based
        
        # Extract all groups of digits
        digit_groups = re.findall(r"\d+", list_str)
        if 0 <= index < len(digit_groups):
            return digit_groups[index]

    # Template 6: Delete/Erase/Remove character from word
    match5 = re.search(r"(?:Delete|Erase|Remove)\s+(?:each|every|all)\s+occurrence(?:s)?\s+of\s+([a-zA-Z])\s+(?:from|in)\s+([a-zA-Z]+)", prompt, re.IGNORECASE)
    if match5:
        char_to_remove = match5.group(1)
        target_word = match5.group(2)
        # Handle case sensitivity by replacing both lower and upper case if needed, though usually exact match
        return target_word.replace(char_to_remove.lower(), "").replace(char_to_remove.upper(), "")

    # Fallback
    return None

if __name__ == "__main__":
    # Test cases
    test_1 = "Only if the ending is 8, replace the last character with 9 in 245666."
    assert solve_hcaptcha_text(test_1) == "245666" # Does not end in 8

    test_2 = "Only if the ending is 8, replace the last character with 9 in 245668."
    assert solve_hcaptcha_text(test_2) == "245669" # Ends in 8
    
    test_3 = "Can you tell me the third set of numeric values in the provided set: 9229 @ 8886 $ 2632 * 382? Please try again. ⚠️"
    assert solve_hcaptcha_text(test_3) == "2632"
    
    test_4 = "Which one is the second collection of numbers within this data string: 5966 * 2521 3897 | 6507 - 2131 - 2023? Please try again. ⚠️"
    assert solve_hcaptcha_text(test_4) == "2521"
    
    test_5 = "Answer the following question with numbers only. Replace the last character with 4 only when the ending character is 8 in 235008. Please try again. ⚠️"
    assert solve_hcaptcha_text(test_5) == "235004"
    
    test_6 = "Please answer the following question only with numbers. Replace the last character with 7 only when the ending character is 9 in 389279. Please try again. ⚠️"
    assert solve_hcaptcha_text(test_6) == "389277"
    
    test_7 = "Please respond to the following question using only numbers. Give the 3rd group of digits from the given list: 6030 - 5913 ... 8144 % 5326 Please try again. ⚠️"
    assert solve_hcaptcha_text(test_7) == "8144"
    
    test_8 = "Can you tell me the starting collection of numbers from this string of numbers: 9669 * 6863 \ 9868 - 8217? Please try again. ⚠️"
    assert solve_hcaptcha_text(test_8) == "9669"
    
    test_9 = "Please provide the initial collection of digits considering this arrangement: 8830 / 1654 + 4862 + 6180 - 6946 Please try again. ⚠️"
    assert solve_hcaptcha_text(test_9) == "8830"
    
    test_10 = "Which one is the fourth set of numerals from this collection of numbers: 6827 \ 7157 * 9377 6732? Please try again. ⚠️"
    assert solve_hcaptcha_text(test_10) == "6732"
    
    test_12 = "Please answer the following question only with letters. Delete each occurrence of r from accurate. Please try again. ⚠️"
    assert solve_hcaptcha_text(test_12) == "accuate"

    test_13 = "Please respond to the following question using only letters. Erase each occurrence of s in congress. Please try again. ⚠️"
    assert solve_hcaptcha_text(test_13) == "congre"

    test_14 = "Please answer the following question only with letters. Delete each occurrence of f from festival. Please try again. ⚠️"
    assert solve_hcaptcha_text(test_14) == "estival"

    test_16 = "Answer the following question with letters only. Delete every occurrence of o in progress. Please try again. ⚠️"
    assert solve_hcaptcha_text(test_16) == "prgress"

    test_17 = "Please respond to the following question using only letters. Remove all occurrences of v from valuable. Please try again. ⚠️"
    assert solve_hcaptcha_text(test_17) == "aluable"
    
    print("All tests passed.")
