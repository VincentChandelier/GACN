
%%%%这个程序是用来计算每个图像的patchsize，因为每个图像的深度不同，所以对应的patch_size也不�?�?
function [ patch_map ] = Patch_Size_Cal( Input_image,Center_map )
%%%
%input:
    %InputImage: Input image that have ratated that all the center of MLA at
        %the same row can located at the same row pixel
    %Center_map: the center map of each MLA
%output:
    %patch_map: the patch size of each MLA

%%%
mm = Center_map;


patch_map=zeros(43,66);%用于存放�?终结�?
gray_image=rgb2gray(Input_image);
%%%%这里只计�?66*43个微透镜的�?�，�?后一行微透镜以上�?行为�?
    for i=1:66
        origin_patch=15;
        for j=1:42
            cor_x=mm(j,i,1);
            cor_y=mm(j,i,2);
            cor_x1=mm(j+1,i,1);
            cor_y1=mm(j+1,i,2);
            if cor_x1*cor_y1~=0
                full_patch=gray_image(cor_y-31:cor_y+31,cor_x-31:cor_x+31,:);
    %             full_patch=imresize(full_patch,10,'bicubic');
    %             patch_1=full_patch(191:290,191:290,:);
                patch_1=full_patch(19:29,19:29,:);
                for k=15:1:33
                    full_patch=gray_image(cor_y1-31:cor_y1+31,cor_x1-31:cor_x1+31,:);
    %                 full_patch=imresize(full_patch,10,'bicubic');
    %                 patch_2=full_patch(191+k:290+k,191:290);
                    patch_2=full_patch(19+k:29+k,19:29,:);
    %                 patch_2=gray_image(cor_y1-boundary+k:cor_y1+boundary+k,cor_x1-boundary:cor_x1+boundary,:);
                    ssim_value(k-14)=ssim(patch_1,patch_2);
                end
                [val,ord]=max(ssim_value);
                if j==1
                    if i==1
                        origin_patch=ord+14;
                    else
                        if ssim_value(patch_map(j,i-1)-14)>0.85
                            origin_patch=patch_map(j,i-1);
                        end
                    end
                else
                    if i==1
                        if ssim_value(origin_patch-14)>0.85 
                           origin_patch=origin_patch;
                        else
                            origin_patch=ord+14;
                        end
                    else
                        if j==42 && mod(i,2)==0
                            if ssim_value(origin_patch-14)>0.85 || ssim_value(patch_map(j-1,i-1)-14)>0.85
                                if ssim_value(origin_patch-14)< ssim_value(patch_map(j-1,i-1)-14)
                                    origin_patch=patch_map(j-1,i-1);
                                else
                                    origin_patch=origin_patch;
                                end
                            else
                                origin_patch=ord+14;
                            end
                        else
                            if ssim_value(origin_patch-14)>0.85 || ssim_value(patch_map(j,i-1)-14)>0.85
                                if ssim_value(origin_patch-14)< ssim_value(patch_map(j,i-1)-14)
                                    origin_patch=patch_map(j,i-1);
                                else
                                    origin_patch=origin_patch;
                                end
                            else
                                origin_patch=ord+14;
                            end
                        end
                    end
                end  
                patch_map(j,i)=origin_patch;
            end
        end
    end
    %
    patch_map(43,:)=patch_map(42,:);