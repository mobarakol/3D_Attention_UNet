clear all;
%addpath('/home/mmlab/Datasets/BraTS18/3D_Reader/mha');
addpath(genpath('/home/navodini/Documents/NUS/Brats19/Feature_Extraction/nifti_utils/'));
addpath('/home/navodini/Documents/NUS/Brats19/Feature_Extraction/boxcount/');
MRI_dir = '/home/navodini/Documents/NUS/Brats19/SegmentedBrats19_test/';

datasets = dir(MRI_dir);

fileID = fopen(['/home/navodini/Documents/NUS/Brats19/test_IDs.txt'],'r');
A=[];
B=[];
for i=1:166
    file = fgetl(fileID);
    new_dir = strcat(MRI_dir,file);
    %img_new_dir = strcat(img_dir,file);
    
    new = strcat(new_dir,'.nii.gz');
    disp(new_dir)
    GT = load_untouch_nii(new);

    GT.img(GT.img==1) = 1;
    GT.img(GT.img== 2) = 0;
    GT.img(GT.img== 4 ) = 0;

    [n, r] = boxcount(GT.img);


    row2 = [n(1:5)];
    B=[B;row2];
    
    
end
csvwrite('fractal_nec.txt',B);
fclose(fileID);
