%%%%This code is used to extract the subaperture of lenslet这个程序使用来做lenslet 图像提取子孔径图像的
function [  ] = Lenslet_Rendering_zoom( Input_image,Center_map,Patch_map, SavePath,ImageName,Views)
%%%We adjust the useage of zoom up to speed the result
%input:
    %InputImage: Input image that have ratated that all the center of MLA at
        %the same row can located at the same row pixel
    %Center_map: the center map of each MLA
    %Patch_map: the PatchMaps of each MLA
    %SavePath: The savedir of the rendered image
    %ImageName: The image name of rendered image folder
    %Views: numbers of views to render

%output:
    %the render images

%%%
    zoom = 3;
    %%%%for adjust zoom up, the patch_map should not to zoom up before the
    %%%%function
    Patch_map = Patch_map*zoom;
    %Do not change
    true_p=20*zoom;
    bound=1*zoom;
    patch_size=true_p+2*bound;
    image=im2double(Input_image);
    mm=Center_map;
    new_patch=ones(patch_size,patch_size,4);
    %             patch_weight=ones(patch_size,patch_size);
    p1=round(true_p/2*sqrt(3));%the width of rendered images 
    %                 p2=round(p/2*sqrt(3));%size of file patch
    Moverange  = 6; %The largest move range on the condition of views of 5. According to the formulation,q = (mR*sqrt(2)-Dmax/2)/((N-1)/2),
    %%We ser m=0.8,R=35,Dmax = 28, thus, q= 2.899,moverange=2*ceil(q)=6
    if Views==1
        t1=0;
        t2=0;
    else
        interval=floor(Moverange*2/(Views-1));
        t1=-Moverange:interval:Moverange;
        t2=-Moverange:interval:Moverange;
    end
    for k1=t1
        for k2=t2
            p=patch_size;
            t=patch_size-p1;
            q=floor(p/2);      %p is large
            q1=floor(true_p/2);
            new_image=zeros(42*true_p+2*bound+q1,66*p1+t,4);%the size of final image
            Grad_diff=0;
            for i=1:66
                for j=1:42
                        if i==63 && j==38
                            cor_x=mm(j-1,i,1);
                            cor_y=mm(j-1,i,2);
                        else             
                            cor_x=mm(j,i,1);
                            cor_y=mm(j,i,2);
                        end
                    %choose the blocks and put it in the final images?
                    if cor_x*cor_y~=0
                        stitch_patch=Patch_map(j,i)+2*bound;
                        q_raw=ceil(Patch_map(j,i)/2/zoom)+3;
                        full_patch=image(cor_y-q_raw+k1:cor_y+q_raw-1+k1,cor_x-q_raw+k2:cor_x+q_raw-1+k2,1:3);
                        full_patch=imresize(full_patch,zoom,'bicubic');   
                        patch=full_patch(3*zoom+1:3*zoom+stitch_patch,3*zoom+1:3*zoom+stitch_patch,:);
                        patch=imresize(patch,[patch_size,patch_size]);
                        new_patch(:,:,1:3)=imrotate(patch,180);

                       if mod(i,2)==1
                            new_image((j-1)*true_p+1:(j-1)*true_p+p,(i-1)*p1+1:(i-1)*p1+p,1:4)=new_image((j-1)*true_p+1:(j-1)*true_p+p,(i-1)*p1+1:(i-1)*p1+p,1:4)+new_patch;
                       else
                            new_image((j-1)*true_p+q1+1:(j-1)*true_p+q1+p,(i-1)*p1+1:(i-1)*p1+p,1:4)=new_image((j-1)*true_p+q1+1:(j-1)*true_p+q1+p,(i-1)*p1+1:(i-1)*p1+p,1:4)+new_patch;
                       end 
                    end
                end
            end
            new_image=new_image(q+1:end-q,:,:);
            weight_matrix=repmat(new_image(:,:,4),1,1,3);
            Final_image=new_image(:,:,1:3)./weight_matrix;
            Final_image=imresize(Final_image,1/zoom);
%             Final_image=imresize(Final_image,[1660,2291]);
            
            Savedir = fullfile(SavePath,ImageName);
            
            if ~exist(Savedir,'dir')
               mkdir(Savedir);
            end
%             imshow(Final_image)
            if Views==1
                save_name=fullfile(Savedir,['central','.png']);
            else
                save_name=fullfile(Savedir,[num2str(floor((k1+Moverange)/interval)+1),'_',num2str(floor((k2+Moverange)/interval)+1),'.png']);
            end
      imwrite(Final_image,save_name);
      clear Final_image new_image weight_matrix
      end
   end
        
    % figure
    % imshow(Final_image);
    % fprintf(sprintf('Grad_diff:%d\n',Grad_diff));
end