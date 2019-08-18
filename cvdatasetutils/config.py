
# Default location of Visual Genome files
VG_OBJECTS = "objects.json"
VG_RELATIONSHIPS = "relationships.json"
VG_ANALYTICS='./analytics'
VG_DATA = './data'
VG_IMAGES='images'
VG_IMAGE_EXTENSION='jpg'

SPLIT_DISTRIBUTION = 0.2
MAX_LOADED_IMAGES = 2000000
TOP_OBJECTS = 1000
GLOBAL_OBJECTS_FILE = 'global_objects.csv'
IMAGE_OBJ_PD_FILE = 'image_objs.csv'
VGOPD_TEST_FILE = 'vgopd_test.csv'
VGOPD_TRAIN_FILE = 'vgopd_train.csv'

DATA_FILES = [
    'https://visualgenome.org/static/data/dataset/objects.json.zip',
    'https://visualgenome.org/static/data/dataset/relationships.json.zip',
    'https://visualgenome.org/static/data/dataset/object_alias.txt',
    'https://visualgenome.org/static/data/dataset/relationship_alias.txt',
    'https://visualgenome.org/static/data/dataset/object_synsets.json.zip',
    'https://visualgenome.org/static/data/dataset/attribute_synsets.json.zip',
    'https://visualgenome.org/static/data/dataset/relationship_synsets.json.zip',
    'https://visualgenome.org/static/data/dataset/image_data.json.zip',
    'https://visualgenome.org/static/data/dataset/region_descriptions.json.zip',
    'https://visualgenome.org/static/data/dataset/question_answers.json.zip',
    'https://visualgenome.org/static/data/dataset/objects_v1_2.json.zip',
    'https://visualgenome.org/static/data/dataset/attributes.json.zip',
    'https://visualgenome.org/static/data/dataset/relationships_v1_2.json.zip',
    'https://visualgenome.org/static/data/dataset/synsets.json.zip',
    'https://visualgenome.org/static/data/dataset/region_graphs.json.zip',
    'https://visualgenome.org/static/data/dataset/scene_graphs.json.zip',
    'https://visualgenome.org/static/data/dataset/qa_to_region_mapping.json.zip'
]

IMAGE_FILES = [
    'https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip',
    'https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip'
]

VG_FOLDER_STRUCTURE = {
    'VG_DATA': VG_DATA,
    'VG_ANALYTICS': VG_ANALYTICS,
    'VG_IMAGES': VG_IMAGES
}

# Pascal VOC Configuration
VOC_IMAGES="JPEGImages"
VOC_ANNOTATIONS="Annotations"






# General configuration
CUDA_DEVICE='0'
STEPS_PER_EPOCH=500
BATCH_LOG_FREQUENCY=50
