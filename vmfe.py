from transformers import AutoImageProcessor, VideoMAEForVideoClassification
import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VideoMAE_FE(nn.Module):
  def __init__(self, model_name='MCG-NJU/videomae-base-finetuned-kinetics'):
    super(VideoMAE_FE, self).__init__()
    self.image_processor = AutoImageProcessor.from_pretrained(model_name)
    self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device)
    self.n_frames = self.model.config.num_frames
  
  def forward(self, videos):
    videos = videos.view(-1, 3, IMG_SIZE, IMG_SIZE).to(device)
    output = self.image_processor(list(videos), return_tensors='pt')
    pixel_values = output['pixel_values']
    new_size = pixel_values.shape[-1]
    pixel_values = pixel_values.view(-1, self.n_frames, 3, new_size, new_size)

    with torch.no_grad():
      result = self.model(pixel_values=pixel_values.to(device))
    
    return result.logits
