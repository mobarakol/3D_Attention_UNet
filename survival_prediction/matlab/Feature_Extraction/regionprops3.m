function output = regionprops3( input, varargin )
% regionprops3 measures the geometric properties of image objects in 
%  3D space. Objects are defined as connected pixels in 3D. This function 
%  uses regionprops to get pixellist from the binary image. If you'd like
%  to define objects connectivity on our own, use bwlabeln first. 
% 
%  output = regionprops3(img,properties) takes 3-D binary image or output 
%  from bwlabeln and returns measurement as specified by properties. If no
%  property is specified, the function will return all measurements by 
%  default.
%
%  output = regionprops3(img,'IsPixList', properties) takes an M x 3 matrix of
%  pixel list as input and returns measurements. 
%  
%  Properties can be a comma-separated list of strings such as: 
% 
%  'MajorAxis' : returns a unit vector that points in the
%  direction of the major axis
%  
%  'MajorAxisLength' : returns the length of the major axis
%
%  'Centroid' : returns the centroid of the object
%
%  'AllAxes' : returns measurements of all three principal axes of image
%   objects, including axis directions, eigenvalues and axis lengths, all
%   organized in descending axis length. 
%  
%  'Eccentricity' : returns Meriodional Eccentricity, defineds as the 
%   eccentricity of the section through the longest and the shortest axes
%   and Equatorial Eccentricity, defined as the eccentricity of the 
%   section through the second longest and the shortest axes. 
%  
%  Version 1.1.1
%  Copyright 2014 Chaoyuan Yeh
    
if any(strcmpi(varargin,'IsPixList'));
    if isstruct(input)
        pixList = input;
    elseif length(size(input))== 2 && size(input,2) == 3;
        pixList.pixList = input;
    else
        error('Pixel list should be either an Mx3 matrix or a structured array of Mx3 matrix');
    end
else
    pixList = regionprops(input, 'PixelList');
end

flag = false;
if numel(varargin)-any(strcmpi(varargin,'IsPixList')) == 0, flag = true; end
if ~isstruct(pixList), pixList.PixelList = pixList; end

for ii = 1:length(pixList)
    pixs = struct2array(pixList(ii));
    covmat = cov(pixs);
    [eVectors, eValues] = eig(covmat);
    eValues = diag(eValues);
    [eValues, idx] = sort(eValues,'descend');
    
    if flag || any(strcmpi(varargin,'MajorAxis')) 
        output(ii).MajorAxis = eVectors(:,idx(1))';
    end
    
    if flag || any(strcmpi(varargin,'MajorAxisLength'))
        distMat = sum(pixs.*repmat(eVectors(:,idx(1))',size(pixs,1),1),2);
        output(ii).MajorAxisLength = range(distMat);
    end
    
    if flag || any(strcmpi(varargin,'AllAxes')) 
        output(ii).FirstAxis = eVectors(:,idx(1))';
        output(ii).SecondAxis = eVectors(:,idx(2))';
        output(ii).ThirdAxis = eVectors(:,idx(3))';
        output(ii).EigenValues = eValues'; 
        distMat = sum(pixs.*repmat(eVectors(:,idx(1))',size(pixs,1),1),2);
        output(ii).FirstAxisLength = range(distMat);
        distMat = sum(pixs.*repmat(eVectors(:,idx(2))',size(pixs,1),1),2);
        output(ii).SecondAxisLength = range(distMat);
        distMat = sum(pixs.*repmat(eVectors(:,idx(3))',size(pixs,1),1),2);
        output(ii).ThirdAxisLength = range(distMat);
    end
    
    if flag || any(strcmpi(varargin,'Centroid')) 
        output(ii).Centroid = mean(pixs,1);
    end
    
    if flag || any(strcmpi(varargin,'Eccentricity'))
        output(ii).MeridionalEccentricity = sqrt(1-(eValues(3)/eValues(1))^2);
        output(ii).EquatorialEccentricity = sqrt(1-(eValues(3)/eValues(2))^2);
    end
end
