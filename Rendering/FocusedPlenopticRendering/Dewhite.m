%%%%This code is used for devignting这个程序使用来做lenslet 图像白图像校正
function [ RecImage ] = Dewhite( Input_image,Center_map)
    %%%
    %input:
        %InputImage: Input image that have ratated that all the center of MLA at
            %the same row can located at the same row pixel
        %Center_map: the center map of each ML

    %output:
        %the rectified image
    %%%
    mm = Center_map;
    load('save_coe.mat');
    Input_image=im2double(Input_image);
    RecImage = Input_image;
    %%devigntings
    for i=1:66
        for j=1:42
                if i==63 && j==38
                    cor_x=mm(j-1,i,1);
                    cor_y=mm(j-1,i,2);
                else             
                    cor_x=mm(j,i,1);
                    cor_y=mm(j,i,2);
                end
    %         cor_x=mm(j,i,1);
    %         cor_y=mm(j,i,2);
            if cor_x*cor_y~=0
                RecImage(cor_y-30:cor_y+30,cor_x-30:cor_x+30,1)=Input_image(cor_y-30:cor_y+30,cor_x-30:cor_x+30,1)./save_coe(:,:,(i-1)*42+j);
                RecImage(cor_y-30:cor_y+30,cor_x-30:cor_x+30,2)=Input_image(cor_y-30:cor_y+30,cor_x-30:cor_x+30,2)./save_coe(:,:,(i-1)*42+j);
                RecImage(cor_y-30:cor_y+30,cor_x-30:cor_x+30,3)=Input_image(cor_y-30:cor_y+30,cor_x-30:cor_x+30,3)./save_coe(:,:,(i-1)*42+j);
            end
        end
    end
    RecImage = cast(RecImage*255,'uint8');