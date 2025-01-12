'''
    function for punctuation removal from string 
    arguments:
        X: string 
    return: 
        string 
'''

def punctuation_removal(text: str) -> str: 
    return "".join([u for u in text if u == ' ' or u == "'" or ord('0') <= ord(u) <= ord('9') or ord('a') <= ord(u) <= ord('z') or ord('A') <= ord(u) <= ord('Z')])