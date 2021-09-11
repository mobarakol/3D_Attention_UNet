clear all;
%addpath('/home/mmlab/Datasets/BraTS18/3D_Reader/mha');
addpath(genpath('/home/navodini/Documents/NUS/Brats19/Feature_Extraction/nifti_utils/'));
MRI_dir = '/home/navodini/Documents/NUS/Brats19/SegmentedBrats19_test/';

datasets = dir(MRI_dir);

fileID = fopen(['/home/navodini/Documents/NUS/Brats19/test_IDs.txt'],'r'); %File containing the names of the testing data
A=[];
B=[];
for i=1:166
    
    file = fgetl(fileID);
    new_dir = strcat(MRI_dir,file);
    
    new = strcat(new_dir,'.nii.gz');
    disp(new_dir);
    GT = load_untouch_nii(new);
    GT.img(GT.img==1) = 1;
    GT.img(GT.img== 2) = 1;
    GT.img(GT.img== 4 ) = 1;

    o = regionprops3( double(GT.img) ); 
    fprintf(fileID,'%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n',...
        new_dir, o.MajorAxis,o.MajorAxisLength,o.FirstAxis,o.SecondAxis,o.ThirdAxis,o.EigenValues,o.FirstAxisLength,...
        o.SecondAxisLength,o.ThirdAxisLength,o.Centroid,o.MeridionalEccentricity,o.EquatorialEccentricity);
    %disp(o.FirstAxis);
    %disp(o.EigenValues);
    %disp(o.FirstAxisLength);
    %disp(o.Centroid);
    row =[o.FirstAxis,o.SecondAxis,o.ThirdAxis,o.EigenValues,o.FirstAxisLength,o.SecondAxisLength,o.ThirdAxisLength,o.Centroid,o.MeridionalEccentricity,o.EquatorialEccentricity];
    A=[A;row];

    
end

csvwrite('geometry_wt.txt',A);
fclose(fileID);