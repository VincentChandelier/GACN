%%%%This code is used to compute the PSNR and MSSSIM of multiviews这个程序计算参考图像和原始图像MultiView之间的PSNR跟MSSSIM
%%%%This program is used to compute the averager PSNR and MS-SSIM銆?of
%%%%multiviews
function [ meanPSNR, meanMSSSIM ] = MultiviewMetrics( InputDir,ImgName,RefDir)
%%%
%input:
    %InputDir: Input image dir
    %ImgName: Input image name
    %RefDir: Ref image dir
  %InputDir
  %%--ImgName.png
      %%--1.png
      %%--2.png
  %RefDirDir
  %%--ImgName.png
      %%--1.png
      %%--2.png

%output:
    %mean_PSNR: average PSNR
    %mean_MSSSIM: averager MSSSIM

%%%
    Imginfo = dir(fullfile(InputDir,ImgName,'*.png'));
    PSNR = zeros(1,length(Imginfo));
    MSSSIM = zeros(1,length(Imginfo));
    for i=1:length(Imginfo)
        MultiImgName = Imginfo(i).name;
        TestImg = imread(fullfile(InputDir,ImgName,MultiImgName));
        RefImg = imread(fullfile(RefDir,ImgName,MultiImgName));
        PSNR(i) = psnr(TestImg,RefImg);
        MSSSIM(i) = mean(multissim(rgb2ycbcr(TestImg),rgb2ycbcr(RefImg)));
    end
   meanPSNR = mean(PSNR);
   meanMSSSIM = mean(MSSSIM);
end