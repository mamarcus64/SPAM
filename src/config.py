"""
File for pointing to specific model checkpoint files.
If running on a local machine, you will need to download the relevant files and set these yourself.
"""

# currently set up for redondo and for CARC
server = 'redondo'

if server == 'CARC':
    original_checkpoint = '/scratch1/mjma/SPAN/interspeech_spam/segment-anything-2/checkpoints/sam2_hiera_tiny.pt'
    config_file = 'sam2_hiera_t.yaml' # relative to segment-anything-2 folder
    spam_checkpoint = '/scratch1/mjma/SPAN/interspeech_spam/spam_finetuning/saved_models/tiny/checkpoints/step_13200.pt'
    unet_file = '/scratch1/mjma/SPAN/interspeech_spam/spam_finetuning/best_unet.pth'
    
elif server == 'redondo':
    original_checkpoint = '/data1/mjma/spam_model_checkpoints/sam2_hiera_tiny.pt'
    config_file = 'configs/sam2_hiera_t.yaml' # relative to segment-anything-2 folder
    spam_checkpoint = '/data1/mjma/spam_model_checkpoints/step_13200.pt'
    unet_file = '/data1/mjma/spam_model_checkpoints/best_unet.pth'
    
else:
    raise Exception(f"Server config for {server} not set up!")
    