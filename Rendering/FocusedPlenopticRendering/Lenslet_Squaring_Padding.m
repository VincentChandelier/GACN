%%%%This code is used to reshape the lenslet images. 
%%%%Since cropping the lenlset images directly would increase the texture
%%%%in the Macroiamge, Here we explore the paading methods by padding the
%%%%nearest pixel
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
    R = 35;%The radius of a macro images
    if Cropsize>2*R+1
        error('Cropsize is no more larger than 71')
    end
    xaixs = ceil(Cropsize/2):Cropsize+1:(Cropsize+1)*66+ceil(Cropsize/2);%%66 whole lenslet is kept in x axis, we extend one pixel in right and bottom of each macroimages
    yaixs = ceil(Cropsize/2):Cropsize+1:(Cropsize+1)*42+ceil(Cropsize/2);%%42 whole lenslet is kept in y axis, we extend one pixel in right and bottom of each macroimages
    [Square_x,Square_y ]= meshgrid(xaixs,yaixs);
    mm = Center_map;
    SquareImg=ones(42*Cropsize,66*Cropsize,3);
    Visited=ones(42*Cropsize,66*Cropsize);
    side = floor(Cropsize/2); % half of the side of each macro images
    SquareCerter = ones(43,67,2);%%%The center of reshape macroimages
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
                Lenslet_patch=Input_image(cor_y-side:cor_y+side+1,cor_x-side:cor_x+side+1,1:3);
                visitedmap = Circle(Lenslet_patch,R );
                SquareImg(Sy-side:Sy+side+1,Sx-side:Sx+side+1,1:3) = Lenslet_patch;%extend one pixel in right and bottom
                Visited(Sy-side:Sy+side+1,Sx-side:Sx+side+1) = visitedmap;
            end
        end
    end
    if ceil((Cropsize-1)/2)>R/sqrt(2)
        %%%%需要做填充的部分，圆内的点Visited 都表示为1，圆外的点Visited表示为0
        %%%%记录圆外点上下最近的位置，从而求均值
        [top_y, bottom_y, left_x, right_x] = found(Visited);
        SquareImg = SquareImg.*Visited;%%直接获得圆内的点
        [height, width, ~] = size(Visited);
        for y = 1:height
            for x = 1:width
                if Visited(y,x)==0
                    %%%%对圆外的点做填充，判断上下左右是否有值
                    if top_y(y,x)+bottom_y(y,x)==0 && left_x(y,x)+right_x(y,x)==0
                        SquareImg(y,x,:)=SquareImg(y-1,x-1,:);
                    elseif top_y(y,x)+bottom_y(y,x)==0
                        SquareImg(y,x,:) = SquareImg(y,x-1,:);
                    elseif left_x(y,x)+right_x(y,x)==0
                        SquareImg(y,x,:) = SquareImg(y-1,x,:);
                    elseif top_y(y,x)~=0 && bottom_y(y,x)~=0 && left_x(y,x)~=0&&right_x(y,x)~=0
                        SquareImg(y,x,:)=1/2*(bottom_y(y,x)*SquareImg(y-top_y(y,x),x,:)...
                            +top_y(y,x)*SquareImg(y+bottom_y(y,x),x,:))/(top_y(y,x)+bottom_y(y,x))+...
                        1/2*(right_x(y,x)*SquareImg(y,x-left_x(y,x),:)...
                            +left_x(y,x)*SquareImg(y, x+right_x(y,x),:))/(right_x(y,x)+left_x(y,x));
                    else%%%由于存在只有一端是参考点的情况
                        SquareImg(y,x,:)=1/2*(top_y(y,x)*SquareImg(y-top_y(y,x),x,:)...
                            +bottom_y(y,x)*SquareImg(y+bottom_y(y,x),x,:))/(top_y(y,x)+bottom_y(y,x))+...
                        1/2*(left_x(y,x)*SquareImg(y,x-left_x(y,x),:)...
                            +right_x(y,x)*SquareImg(y, x+right_x(y,x),:))/(right_x(y,x)+left_x(y,x));
                    end
                end
            end
        end
        SquareImg = cast(SquareImg,'uint8');
    else
        SquareImg = cast(SquareImg,'uint8');
    end
end
    
function visitedmap = Circle(Macropatch,R )
%%% Remove pixels outside the circle
%input:
    %patch: Input macroimage patch
    %R: radius of the macro image
%output:
    %visitedmap:visited maps
    [height, width, channel] = size(Macropatch);
    center_x = floor(width/2);
    center_y = floor(height/2);
    visitedmap = zeros(height, width);
    for y = 1:height
        for x = 1: width
            if (y-center_y)^2+(x-center_x)^2<=R^2
                visitedmap(y,x) = 1;
            end
        end
    end
end

function [top_y, bottom_y, left_x, right_x] = found(visited)
    [height, width, ~] = size(visited);
    top_y = zeros(height, width);
    bottom_y = zeros(height, width);
    left_x = zeros(height, width);
    right_x = zeros(height, width);
    Range = 70;%%%上下左右搜素范围
    for y = 1:height
        for x = 1:width
            if visited(y,x)==0
                %%%%上端最近
                for up = 1:Range
                    if y-up<1
                        break
                    elseif visited(y-up,x)==1
                        top_y(y,x) = up;
                        break
                    end
                end
                %%%%下端最近
                for down = 1:Range
                    if y+down>height
                        break
                    elseif visited(y+down,x)==1
                        bottom_y(y,x) = down;
                        break
                    end
                end
                %%%%左端最近
                for left = 1:Range
                    if x-left<1
                        break
                    elseif visited(y,x-left)==1
                        left_x(y,x) = left;
                        break
                    end
                end
                %%%%右端最近
                for right = 1:Range
                    if x+right>width
                        break
                    elseif visited(y,x+right)==1
                        right_x(y,x) = right;
                        break
                    end
                end
                
            end
        end
    end
end

                 
                
    
         
    
        
        
    