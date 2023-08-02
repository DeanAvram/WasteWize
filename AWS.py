import requests
from PIL import Image
import io

from torchvision.models import ResNet50_Weights

import Network
import torch
import boto3
import warnings

# warnings.filterwarnings("ignore", category=UserWarning)

region_name = 'eu-central-1'
bucket_name = 'wasteimagestoclassify'

s3 = boto3.client('s3', region_name=region_name)


def predict_from_aws(bucket, path):
    # Get object from bucket and predict
    try:
        response = s3.get_object(Bucket=bucket, Key=path)
        image_data = response['Body'].read()
        stream = io.BytesIO(image_data)
        image = Image.open(stream)

        device = Network.get_default_device()
        model = Network.ResNet()
        Network.to_device(model, device)

        model.load_state_dict(torch.load("model_file.pt", map_location=torch.device('cpu')))
        model.eval()

        print(Network.predict_external_image(model, image))
    except:
        print("An exception occurred")


predict_from_aws(bucket_name, 'folder1/1.jpg')


# Upload object to bucket
def upload_photo_to_aws(bucket, folder_name):

    file_path = "img2.jpg" # can be replaced with photo from http request
    s3.put_object(Bucket=bucket, Key=folder_name)

    object_key = 'folder1/testImg.jpg'
    s3.upload_file(file_path, bucket_name, object_key)


upload_photo_to_aws(bucket_name, "folder1")

