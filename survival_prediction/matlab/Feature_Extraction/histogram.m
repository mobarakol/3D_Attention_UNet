clear all;

%addpath('/home/mmlab/Datasets/BraTS18/3D_Reader/mha');
addpath(genpath('/home/navodini/Documents/NUS/Brats19/Feature_Extraction/nifti_utils/'));
MRI_dir = '/home/navodini/Documents/NUS/Brats19/SegmentedBrats19_test/';

MRI_datasets = dir(MRI_dir);
fileID = fopen(['/home/navodini/Documents/NUS/Brats19/test_IDs.txt'],'r');
A=[];

B=[];

for i=1:166
    i;

    vol=0;
    kurt=0;
    ent=0;
    hist=0;
    file = fgetl(fileID);
    new_dir = strcat(MRI_dir,file);
        
    new = strcat(new_dir,'.nii.gz');
    disp(new);

    GT = load_untouch_nii(new);


    GT.img(GT.img==1) = 0;

    GT.img(GT.img== 2) = 0;

    GT.img(GT.img== 4 ) = 1;
        
    for m=1:155
        
        hist=hist+sum(extractHOGFeatures(GT.img(:,:,m)));
        
    end
    hist=hist/1000;
 
    kurt = kurtosis(kurtosis(kurtosis(double(GT.img))));

    ent = entropy(double(GT.img));

    row2 =[kurt ent hist ];

    B=[B;row2]; 

end

csvwrite('hist_enh.txt',B);
fclose(fileID);