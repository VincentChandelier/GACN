%%% This file is to rendering sub-aperture images from preprocessed focused
%%% plenoptic images

clc;
clear all;
PerFolderFiles = [];
AllFiles = [];
FocusedPlenopticRenderingPath='.\FocusedPlenopticRendering';
addpath(FocusedPlenopticRenderingPath);
DefaultFileSpec = {'*.png','*.bmp'};
InputPath = {'.\PreprocessedImages'}; % focused plenoptic image dictory
%---Defaults---
DefaultPath = '.\PreprocessedImages';
OutPath = {'.\PreprocessedSAIs'};
%%%get the files from the focused plenoptic image dictory
[FileList, BasePath] = FindFilesRecursive( InputPath, DefaultFileSpec, DefaultPath );
Views = 5; %number of Views
%%%the specific type 
% Lenset_ori: original lenslet images.
% Lenslet_insqur: The rearanged insquare lenslet images
Mode = "Lenslet_insqur";
for( iFile = 1:length(FileList) ) %length(FileList)
    CurFname = FileList{iFile};
    Impath = fullfile(BasePath, CurFname);
    ImageName = CurFname(1:end-4);
    SquareImg =imread(Impath);
    load('InSquareCerter.mat');
    load(['.\PatchMap\',ImageName,'_patchmaps']);
    Lenslet_Rendering( SquareImg,SquareCerter,patch_map, fullfile(OutPath{1},Mode), ImageName,Views)
end
