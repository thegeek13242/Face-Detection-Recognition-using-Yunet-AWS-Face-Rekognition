import boto3
import face_collections as fcol

s3 = boto3.client('s3')

COL_NAME: str = 'chef-faces'  # Name of the collection
BUCKET_NAME: str = 'analytics-124' # Name of the s3 bucket

if __name__ == '__main__':
    fcol.create_collection(COL_NAME)
    filenames = []
    for key in s3.list_objects(Bucket=BUCKET_NAME)['Contents']:
            filenames.append(key['Key'])

    for file in filenames:
            if('/' in file):
                continue
            fcol.add_faces_to_collection(BUCKET_NAME,file,COL_NAME)

    print("Collection Initialized!")
