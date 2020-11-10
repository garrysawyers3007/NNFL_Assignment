import torch
from PIL import Image
from torchvision import transforms
from GoogleNet import GoogleNet

def load_google_net():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'googlenet', pretrained=True)
    model = GoogleNet(model)# extract from last 1024 dim layer
    model.eval()
    return model

def load_fcn():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'fcn_resnet101', pretrained=True)
    model.eval()
    return model

def get_features_from_google_net(model, filename, device):
    model.eval()
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)

    input_batch = input_batch.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(input_batch)

    return output

def get_fcn_output(model, filename, device):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to(device)
        model.to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    return output_predictions

def create_glove_dict(path_to_text:str):
    """
    Create the dictionary containing word and corresponding vector. 
    :param path_to_text: Path to Glove embeddings.  
    """
    embeddings = {}
    '''
    Your code goes here. 
    '''
    with open(path_to_text, 'r', encoding='utf-8') as f:
        for line in f:
            val=line.split(' ')
            word = val[0]
            vector = [float(x) for x in val[1:]]
            embeddings[word]=vector
    return embeddings

def loss_func(y_t, alpha, beta, p=2, q=0.5):
    pass