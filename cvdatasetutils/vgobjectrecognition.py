from torch.utils.data import Dataset
import cvdatasetutils.visualgenome as vg
import cvdatasetutils.config as conf
from mltrainingtools.cmdlogging import section_logger
import os


class VGObjectRecognition(Dataset):
    def __init__(self, dataset_folder, images_folder, test=False):
        return None

    def load_data(self, dataset_folder):
        return None


    def __len__(self):
        return None


    def __getitem__(self, idx):
        return None


def extract_region_information(data):
    return data


def generate_rp_from_vg(output_path, input_path, perc):
    vg.set_base(input_path)
    section = section_logger()

    section('Loading Visual Genome')
    data = vg.load_visual_genome(os.path.join(input_path, conf.VG_DATA))

    section('Extracting Image Region Information')
    regions = extract_region_information(data)



if __name__ == "__main__":
    generate_rp_from_vg("./data", "", 0.2)