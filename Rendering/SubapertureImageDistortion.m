%%%%end-to-end decoded image to multiview and compute the PSNR这里需要将端到端图形解码图像，进行转化为multiview，并计算PSNR
%%%% Let the decoder png to multiview, and compute
%%%% PSNR
clc;
clear all;
FocusedPlenopticRenderingPath='.\FocusedPlenopticRendering';
addpath(FocusedPlenopticRenderingPath);
DefaultFileSpec = {'*.png'};
InputPath = '.\TestResults\End2EndResults';%%End2end compression results
%---Defaults---
DefaultPath = '.\TestResults\End2EndResults';
OutPath = '.\TestResults\End2endMultiviewResult';
Ref = '.\OriginalSAIs';
dirpath = dir(InputPath);
for (dirnum = 3:length(dirpath))
    Method = dirpath(dirnum).name;
    [FileList, BasePath] = FindFilesRecursive( [InputPath,'\',Method], DefaultFileSpec, [DefaultPath,'\',Method] );%%%get the files of 
    %%%the specific type 
    % Lenset_ori: original lenslet images.Lenslet_outsqur: the rearanged
    % outsquare lenslet images
    % Lenslet_insqur: The rearanged insquare lenslet images
    Views = 5; %number of Views
    Mode = "Lenslet_insqur";
    %%%get the name of method
    RefDir = fullfile(Ref,"Lenset_ori");
    if exist(fullfile(OutPath,Method,'Result.csv'),'file')
        delete(fullfile(OutPath,Method,'Result.csv'));
    end
    if ~exist(fullfile(OutPath,Method),'dir')
        mkdir(fullfile(OutPath,Method));
    end
    fid = fopen(fullfile(OutPath,Method,'Result.csv'),'a+');
    fprintf(fid,'%s,%s,%s\n','ImageName','PSNR','MSSSIM');
    for( iFile = 1:length(FileList) )
        CurFname = FileList{iFile};
        Imgpath = fullfile(BasePath, CurFname);
        idx=strfind(CurFname(1:end),'\');
        if isempty(idx)>0
            ImageName = CurFname(1:end-4);%%%original Name is Name444.yuv
            folder = CurFname(1:end-4);
        else
            ImageName = CurFname(idx(end)+1:end-4);
            folder = CurFname(1:idx(1)-1);
        end
        %%%save the multiview images
        tempfile = fullfile(OutPath,Method);
        if ~exist(tempfile,'dir')
            mkdir(tempfile)
        end
        if Mode ==  "Lenset_ori"
            width = 4080;
            height = 3068;
        elseif Mode ==  "Lenslet_insqur"
            width = 3168;
            height = 2016;
        end
        Image =imread(Imgpath);
        if Mode ==  "Lenset_ori"
            load('LenseltCenter.mat');
            Image = imrotate(Image,rotAngle,'bilinear');
            Center_map=cast(cast(mm_rotated,'uint16'),'double');
            %%%Devignting
            Image = Dewhite(Image,Center_map);
        end
        load(['PatchMap\',ImageName,'_patchmaps']);
        if Mode ==  "Lenset_ori"
            Lenslet_Rendering( Image,Center_map,patch_map, fullfile(tempfile,Mode), ImageName,Views)
        elseif Mode ==  "Lenslet_insqur"
            load('InSquareCerter','SquareCerter');
            Lenslet_Rendering( Image,SquareCerter,patch_map, fullfile(tempfile,Mode), ImageName,Views)
        end

        [meanPSNR, meanMSSSIM ] = MultiviewMetrics( fullfile(tempfile,Mode),ImageName, RefDir);
        fprintf(fid,'%s,%s,%s\n',ImageName,num2str(meanPSNR),num2str(meanMSSSIM));
    end
    fclose(fid);
end