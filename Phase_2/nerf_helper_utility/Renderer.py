import torch
import torch.nn.functional as F

def render_function(radiance, origin_ray, depth_val):
  #Extract the channel from radiance field tensor and pass it to ReLU activation function
  sigma = F.relu(radiance[...,3])   

  #Apply sigmoid function to the radiance tensor
  x = torch.sigmoid(radiance[...,:3])

  expo = torch.tensor([1e10], dtype = origin_ray.dtype, device = origin_ray.device)

  concatenated_distribution = torch.cat((depth_val[...,1:] - depth_val[...,:-1], expo.expand(depth_val[...,:1].shape)), dim = -1)

  y = 1. - torch.exp(-sigma * concatenated_distribution)

  tensor_weight = y * cumulative_product(1. - y + 1e-10)

  rgb_Data = (tensor_weight[..., None] * x).sum(dim = -2)

  depth_Data = (tensor_weight * depth_val).sum(dim = -1)

  acc_Data = tensor_weight.sum(-1)

  return rgb_Data, depth_Data, acc_Data

#Compute Cumulative product of tensors
def cumulative_product(tensor):

  #Calculate the cumulative product of tensor and shift the tensor by 1 place towards left
  cum_product = torch.cumprod(tensor, dim=-1)
  cum_product = torch.roll(cum_product, 1, dims=-1)
  cum_product[..., 0] = 1.
  
  return cum_product