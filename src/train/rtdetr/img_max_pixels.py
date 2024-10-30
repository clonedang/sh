import os
import PIL


# Get the path to the PIL module
PIL_PATH = os.path.dirname(PIL.__file__)
IMAGE_PY = os.path.join(PIL_PATH, 'Image.py')


# Open Image.py and modify MAX_IMAGE_PIXELS
try:
    with open(IMAGE_PY, 'r') as f:
        lines = f.readlines()
        
    # Find the line with MAX_IMAGE_PIXELS and modify it
    for i, line in enumerate(lines):
        if line.startswith('MAX_IMAGE_PIXELS'):
            lines[i] = 'MAX_IMAGE_PIXELS = None\n'
            break
    
    # Write back the modified lines to Image.py
    with open(IMAGE_PY, 'w') as f:
        f.writelines(lines)
        
    print('MAX_IMAGE_PIXELS set to None successfully in Image.py.')

except FileNotFoundError:
    print(f'Image.py not found in {PIL_PATH}. Please check your PIL installation path.')
except Exception as e:
    print(f'Error occurred while modifying Image.py: {str(e)}')