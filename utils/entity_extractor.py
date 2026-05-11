"""
Entity Extractor for Parameterized Bash Command Generation
-----------------------------------------------------------
Problem: The seq2seq model only knows tokens seen in training.
Solution: Replace named entities with typed placeholders BEFORE inference,
          run the model, then substitute the real values back.
 
Example:
  Input:  "make a c file named bhagya"
  →Step1: "make a <EXT> file named <NAME>"   (entity extraction)
  →Step2: "touch <NAME>.<EXT>"               (model inference on canonical form)
  →Step3: "touch bhagya.c"                   (entity reinjection)
"""
 
import re
from dataclasses import dataclass, field
from typing import Optional

# supported file extenstions for now

FILE_EXTENSIONS = {
    'c': 'c', 'python': 'py', 'py': 'py', 'javascript': 'js', 'js': 'js',
    'typescript': 'ts', 'ts': 'ts', 'java': 'java', 'cpp': 'cpp', 'c++': 'cpp',
    'go': 'go', 'rust': 'rs', 'html': 'html', 'css': 'css', 'text': 'txt',
    'txt': 'txt', 'json': 'json', 'yaml': 'yaml', 'yml': 'yml', 'sh': 'sh',
    'bash': 'sh', 'markdown': 'md', 'md': 'md', 'ruby': 'rb', 'php': 'php',
}


@dataclass
class ExtractedEntities:
    name: Optional[str] = None          # e.g."bhagya", "project"
    extension: Optional[str] = None     # e.g."c", "py"
    filename: Optional[str] = None      # e.g."bhagya.c" (already has extension)
    directory: Optional[str] = None     # e.g."backup", "src/"
    url: Optional[str] = None           # e.g."https://example.com"
    pattern: Optional[str] = None       # e.g."*.py", "error"
    number: Optional[str] = None        # e.g."10", "5"

def extract_entities(sentence: str) -> tuple[str, ExtractedEntities]:
    """
    Extract named entities from a natural language sentence and replace them withe a 
    typed placeholders.
    
    returns: (canonical_sentence, ExtractedEntities)
    
    """

    entities = ExtractedEntities()
    processed = sentence.lower().strip()

    # Extract explicit filenames
    filename_match = re.search(r'\b(?P<base>\w+)\.(?P<ext>\w{1,5})(?=\s|$)', processed)
    if filename_match: 
        full_name = filename_match.group(0)
        entities.filename = full_name
        entities.name = filename_match.group('base')  # name without extension
        entities.extension = filename_match.group('ext')  # extension
        processed = processed.replace(full_name, '<FILENAME>')
    
    # extract "named <word>" or "called <word> "

    if not entities.name:
        name_match = re.search(r'(?:named?|called)\s+(\w+)', processed)
        if name_match:
            entities.name = name_match.group(1)
            processed = re.sub(r'(?:named?|called)\s+\w+',
                               'named <NAME>', processed)
    
    # Extract file extension from phrase like  " a python file", " a c file"
    if not entities.extension:
        for lang, ext in FILE_EXTENSIONS.items():
            # match " a pythopn file " or "python file"
            pattern = rf'\b{re.escape(lang)}\s+file\b'
            if re.search(pattern, processed):
                entities.extension = ext
                processed = re.sub(pattern, '<EXT> file', processed)
                break
    
    # Extract directory/folder name
    if not entities.directory:
        dir_match = re.search(
            r'(?:directory|folder|dir)\s+(?:called|named)\s+(\w+)', processed
        )
        if dir_match:
            entities.directory = dir_match.group(1)
            processed = re.sub(
                r'(?:directory|folder|dir)\s+(?:called|named)\s+\w+',
                'directory named <DIRNAME>', processed
            )
        else:
            # "to <dirname> folder" or "into <dirname>/"
            dir_match2 = re.search(r'\bto\s+(\w+)(?:\s+folder|/)', processed)
            if dir_match2:
                entities.directory = dir_match2.group(1)
                processed = re.sub(r'\bto\s+\w+(?:\s+folder|/)',
                                   'to <DIRNAME>/', processed)
    
    # Extract numbers
    if not entities.number:
        num_match = re.search(r'\b\d+\b', processed)
        if num_match:
            entities.number = num_match.group(0) # used to be 1 because 1 causes issues with "show last 10 lines" → "show last <NUM> lines" but then reinjection fails because of multiple numbers in the sentence. Now we replace only the first occurrence of a number with <NUM> and leave the rest as is.
            processed = re.sub(r'\b\d+\b', '<NUM>', processed, count=1)
    return processed, entities

def reinject_entities(command: str, entities: ExtractedEntities) -> str:
    """
    Substitute placeholder tokens in the model output with real entity values.
 
    Example:
      command  = "touch <NAME>.<EXT>"
      entities = ExtractedEntities(name="bhagya", extension="c")
      returns  = "touch bhagya.c"
    """
    result = command
 
    if '<FILENAME>' in result and entities.filename:
        result = result.replace('<FILENAME>', entities.filename)
 
    if '<NAME>' in result and entities.name:
        result = result.replace('<NAME>', entities.name)
 
    if '<EXT>' in result and entities.extension:
        result = result.replace('<EXT>', entities.extension)
 
    if '<DIRNAME>' in result and entities.directory:
        result = result.replace('<DIRNAME>', entities.directory)
 
    if '<NUM>' in result and entities.number:
        result = result.replace('<NUM>', entities.number)
 
    # Handle composite: touch <NAME>.<EXT> when only name or ext is available
    if '<NAME>' in result and entities.filename:
        result = result.replace('<NAME>', entities.filename.split('.')[0])
    if '<EXT>' in result and entities.filename and '.' in entities.filename:
        result = result.replace('<EXT>', entities.filename.split('.')[-1])
    
    #  Fallback: model output "touch <NAME>" instead of "touch <NAME>.<EXT>" 
    if entities.name and entities.extension:
        if entities.name in result and f'.{entities.extension}' not in result:
            result = result.replace(entities.name, f'{entities.name}.{entities.extension}', 1)


    return result
 
 
def process_command(sentence: str, model_inference_fn) -> str:
    """
    Full pipeline: extract → infer → reinject.
 
    Args:
        sentence: Raw natural language input from user.
        model_inference_fn: Callable that takes a (canonicalized) sentence
                            and returns a bash command string with placeholders.
    Returns:
        Final bash command with real values substituted in.
    """
    canonical, entities = extract_entities(sentence)
    raw_command = model_inference_fn(canonical)
    final_command = reinject_entities(raw_command, entities)
    return final_command


# Quick test
if __name__ == "__main__":
    tests = [
        "make a c file named bhagya",
        "create a python file called main",
        "remove file data.csv",
        "move project.txt to backup folder",
        "show last 20 lines of log.txt",
        "create a new directory called src",
        "find all javascript files",
    ]
 
    for t in tests:
        canonical, ents = extract_entities(t)
        print(f"Input:     {t}")
        print(f"Canonical: {canonical}")
        print(f"Entities:  {ents}")
        print()
 