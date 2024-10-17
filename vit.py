import logging
import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig


logger = logging.getLogger(__file__)


class VIT(nn.Module):
    def __init__(self, args, device):
        super(VIT, self).__init__()
        self.config = ViTConfig.from_pretrained(args.vit_path, output_attentions=True)
        self.vit = ViTModel.from_pretrained(args.vit_path, config=self.config)
        self.device = device
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(1024, args.num_classes)
        self.epsilon = 1e-10
        self.mask_portion = args.mask_portion

    
    def forward(self, x, ig_features=None, flag_train=False):
        
        vit_out = self.vit(x)
        vit_output = vit_out.pooler_output#last_hidden_state
        labels = self.classifier(vit_output)
        
        
        #attentions = vit_out.attentions #[-1] 
        attentions = vit_out.attentions[-1] #torch.Size([64, 16, 197, 197]) last_layer
        average_attentions_over_layers = attentions.mean(dim=1) 
        average_attentions_samples = average_attentions_over_layers.mean(dim=1)
        
        
        #labels = self.get_output(output)
        exp_losses = 0.0
        if flag_train:
            for index in range(0, len(x)):# type: ignore
                #index = torch.randint(0, len(seq_list1), (1,))
                x = torch.tensor(x, dtype=torch.float64).to(self.device)
                average_attentions_samples = average_attentions_samples.to(self.device)
                
                sample_average_attentions = average_attentions_samples[index]

                ig_feature = torch.tensor(ig_features[index], dtype=torch.float64).to(self.device)# type: ignore
                
                random_indices = torch.randperm(len(ig_feature))
                top_k_percent = int(self.mask_portion * len(random_indices))
                top_indices = random_indices[:top_k_percent]

                
                mask_attributions = torch.ones_like(sample_average_attentions)
                mask_attributions[top_indices] = 0
                masked_sample_token_attentions = sample_average_attentions * mask_attributions

                mask_ig = torch.ones_like(ig_feature)
                mask_ig[top_indices] = 0
                masked_ig_feature = ig_feature * mask_ig

                
                #abs_diff = self.min_max_normalization(masked_sample_token_attentions) - self.min_max_normalization(masked_ig_feature)
                abs_diff = torch.abs(self.min_max_normalization(masked_sample_token_attentions) - self.min_max_normalization(masked_ig_feature))
                sorted_diff, _ = torch.sort(abs_diff, descending=True)

                num_elements = int(0.5 * sorted_diff.numel())#1.0 * 
                top_20_percent_values = sorted_diff[:num_elements]
                exp_losses += top_20_percent_values.mean(dim=0)
                
            return labels, exp_losses
           
        else:
            return labels, average_attentions_samples
        

    def min_max_normalization(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        normalized_tensor = (tensor - min_val) / (max_val - min_val + self.epsilon)
        return normalized_tensor
        