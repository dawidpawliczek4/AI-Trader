from typing import List

def longest_sequence(sequences: List[str]) -> int:
    '''
        function for finding lenght of the longest sequence
        intput:
            sequences: List[str]
        return:
            int
    '''
    
    return max([len(sequence.split()) for sequence in sequences]) 