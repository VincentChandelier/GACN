%%%%This code is used to reshape the lenslet images
function [ SquareImg, SquareCerter ] = Lenslet_Squaring( Input_image,Center_map, Cropsize)
%%%
%input:
    %InputImage: Input image that have ratated that all the center of MLA at
        %the same row can located at the same row pixel
    %Center_map: the center map of each MLA
    %Cropsize: the size of crop
%output:
    %lenslet square image
    %The Center of each square macro image 
    
    %The size of square lenslet images
%     xaixs = ceil(Cropsize/2):Cropsize:Cropsize*66+ceil(Cropsize/2);%%66 whole lenslet is kept in x axis?
%     yaixs = ceil(Cropsize/2):Cropsize:Cropsize*42+ceil(Cropsize/2);%%42 whole lenslet is kept in y axis?
    if mod(Cropsize,2)
        xaixs = ceil(Cropsize/2):Cropsize:(Cropsize)*66+ceil(Cropsize/2);%%66 whole lenslet is kept in x axis, we extend one pixel in right and bottom of each macroimages
        yaixs = ceil(Cropsize/2):Cropsize:(Cropsize)*42+ceil(Cropsize/2);%%42 whole lenslet is kept in y axis, we extend one pixel in right and bottom of each macroimages
    else
        xaixs = ceil(Cropsize/2):Cropsize:(Cropsize)*66+ceil(Cropsize/2);%%66 whole lenslet is kept in x axis, we extend one pixel in right and bottom of each macroimages
        yaixs = ceil(Cropsize/2):Cropsize:(Cropsize)*42+ceil(Cropsize/2);%%42 whole lenslet is kept in y axis, we extend one pixel in right and bottom of each macroimages
    end
    [Square_x,Square_y ]= meshgrid(xaixs,yaixs);
    mm = Center_map;
    SquareImg=ones(42*Cropsize,66*Cropsize,3);
    side = floor(Cropsize/2); % half of the side of each macro images
    SquareCerter = ones(43,67,2);
    SquareCerter(:,:,1) = Square_x;
    SquareCerter(:,:,2) = Square_y;
    for i=1:66
        for j=1:42
                if i==63 && j==38
                    cor_x=mm(j-1,i,1);
                    cor_y=mm(j-1,i,2);
                else             
                    cor_x=mm(j,i,1);
                    cor_y=mm(j,i,2);
                end
            %get block
            Sx = Square_x(j,i);
            Sy = Square_y(j,i);
            if cor_x*cor_y~=0
                if mod(Cropsize,2)
                    Lenslet_patch=Input_image(cor_y-side:cor_y+side,cor_x-side:cor_x+side,1:3);
                    SquareImg(Sy-side:Sy+side,Sx-side:Sx+side,1:3) = Lenslet_patch;%extend one pixel in right and bottom
                else
                    Lenslet_patch=Input_image(cor_y-side+1:cor_y+side,cor_x-side+1:cor_x+side,1:3);
                    SquareImg(Sy-side+1:Sy+side,Sx-side+1:Sx+side,1:3) = Lenslet_patch;%extend one pixel in right and bottom
                end
            end
        end
    end
    SquareImg = cast(SquareImg,'uint8');