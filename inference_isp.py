from data.demo2 import use_rawpy

if __name__ == '__main__':
    # test = np.load('./data_raw_npz/a0004-jmac_MG_1384.npz')
    images_dir = 'D://DNG//'
    output_dir = 'D://DNG//'
    # down(images_dir, output_dir)
    use_rawpy(images_dir, output_dir)