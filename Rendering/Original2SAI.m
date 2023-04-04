%%% This file is to rendering sub-aperture images from original focused
%%% plenoptic images

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
OutPath = {'.\OriginalSAIs'}; % sub-aperture images dictory from Original focused plenoptic image
%%%get the files from the focused plenoptic image dictory
[FileList, BasePath] = FindFilesRecursive( InputPath, DefaultFileSpec, DefaultPath );
Views = 5; %number of Views
%%%the specific type 
% Lenset_ori: original lenslet images.
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
          %According to Guotai, the max depth is no more than 28=mR=0.8*35
        save(['.\PatchMap\',ImageName,'_patchmaps'],'patch_map');
    else
        load(['.\PatchMap\',ImageName,'_patchmaps']);
    end
    Lenslet_Rendering( Image,Center_map,patch_map, fullfile(OutPath{1},Mode), ImageName,Views)
end
