%%% This file is to preprocess the original focused plenoptic image to the
%%% preprocess focused plenoptic images

clc;
clear all;
PerFolderFiles = [];
AllFiles = [];
FocusedPlenopticRenderingPath='.\FocusedPlenopticRendering';
addpath(FocusedPlenopticRenderingPath);
DefaultFileSpec = {'*.png','*.bmp'};
InputPath = {'.\Test'}; % Original focused plenoptic image dictory
%---Defaults---
DefaultPath = '.\Test';
OutPath = '.\PreprocessedImages';
if ~exist(OutPath,'dir')
   mkdir(OutPath);
end
%%%get the files from the focused plenoptic image dictory
[FileList, BasePath] = FindFilesRecursive( InputPath, DefaultFileSpec, DefaultPath );
Mode = "Lenset_ori";
for( iFile = 1:length(FileList) ) %length(FileList)
    CurFname = FileList{iFile};
    Impath = fullfile(BasePath, CurFname);
    ImageName = CurFname(1:end-4);
    Image =imread(Impath);
    load('LenseltCenter.mat');
    Image = imrotate(Image,rotAngle,'bilinear');
    Center_map=cast(cast(mm_rotated,'uint16'),'double');
    %%%Devignting
    Image = Dewhite(Image,Center_map);
    if ~exist(['.\PatchMap\',ImageName,'_patchmaps.mat'],'file')
        patch_map = Patch_Size_Cal(Image,Center_map);
        patch_map(patch_map>28) = 28; 
          %According to Guotai, the max depth is no more than 28
        save(['.\PatchMap\',ImageName,'_patchmaps'],'patch_map');
    else
        load(['.\PatchMap\',ImageName,'_patchmaps']);
    end
    %%% preprocess the original focused plenoptic images to preprocessed
    %%% focused plenoptic images
    [SquareImg, SquareCerter]  = Lenslet_Squaring( Image,Center_map, 48);
    imwrite(SquareImg,fullfile(OutPath,[ImageName,'.png']));
%     save('InSquareCerter','SquareCerter');
end
