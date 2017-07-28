function show_pixel_anno

filenames = dir('annotations/pixel-level');
BB_file_train = fopen('coparser_train_bbx_gt.txt','w');
BB_file_val = fopen('coparser_val_bbx_gt.txt','w');
SIZE_file = fopen('test_name_size.txt','w');
LABEL_file = fopen('labelmap_voc.prototxt','w')

load('label_list', 'label_list'); % load label list
export_labels(LABEL_file,label_list)

colors = colormap( jet(length(label_list)) );   % set color map

for i = 1: length(filenames),
    %imname = '0006';
    if strcmp(filenames(i).name,'.') || strcmp(filenames(i).name, '..'),
        continue;
    end
    imname = filenames(i).name
    imname = imname(1:end-4);

    load(['annotations/pixel-level/' imname '.mat'], 'groundtruth'); % load an pixel-level annotation

    impath_train = ['JPEGImages/trainval/' imname '.jpg'];
    impath_test = ['JPEGImages/test/' imname '.jpg'];
    %impath = [imname '.jpg'];
    if sum(groundtruth(:))==0,
        continue;
    end

    try
        im = imread(impath_train);    % train
        fprintf(BB_file_train,'%s\n',[imname '.jpg']);
        fprintf(SIZE_file,'%s %d %d\n',imname,size(im,1),size(im,2));
        show_anno(label_list, groundtruth, im, colors, BB_file_train, false);
    catch
        im = imread(impath_test); %test
        fprintf(BB_file_val,'%s\n',[imname '.jpg']);
        fprintf(SIZE_file,'%s %d %d\n',imname,size(im,1),size(im,2));
        show_anno(label_list, groundtruth, im, colors, BB_file_val, false);
    end


end
fclose(SIZE_file);
fclose(BB_file_train);
fclose(BB_file_val);

end


function export_labels(LABEL_file,label_list)
    fprintf(LABEL_file,'item {\n');
    fprintf(LABEL_file,'  name: \"none_of_the_above\"\n');
    fprintf(LABEL_file,'  label: 0\n');
    fprintf(LABEL_file,'  display_name: \"background\"\n');
    fprintf(LABEL_file,'}\n');

    for idx = 2:length(label_list),
        fprintf(LABEL_file,'item {\n');
        fprintf(LABEL_file,'  name: \"%s\"\n', label_list{idx});
        fprintf(LABEL_file,'  label: %d\n', idx-1);
        fprintf(LABEL_file,'  display_name: \"%s\"\n',label_list{idx});
        fprintf(LABEL_file,'}\n');
    end

end



function show_anno(label_list, groundtruth, im, colors, BB_file, b_display)


% get image-level labels name
cur_labels = unique(groundtruth);
label_names = cell(1, length(cur_labels));
for i = 1:length(cur_labels)
    label_names(i) = label_list( cur_labels(i)+1 );
end

if b_display == true,
    f = figure;
    % % 1. show original photo
    subplot(1, 3, 1);  imshow(im); hold on; title('Original'); 
end


% % 2. visualize annotation
gt_image = zeros(size(groundtruth, 1), size(groundtruth, 2), 3);

for labelidx = 1:length(cur_labels)
    if cur_labels(labelidx) == 0,
        continue;
    end
    fprintf(BB_file,'%d\n',cur_labels(labelidx));
    [rows cols] = find(groundtruth == cur_labels(labelidx));

    curcolor = colors(cur_labels(labelidx)+1, :);

    
    obj_image = zeros(size(groundtruth, 1), size(groundtruth, 2), 3);
    for i=1:length(rows)
        obj_image(rows(i), cols(i), 1) = curcolor(1);
        obj_image(rows(i), cols(i), 2) = curcolor(2);
        obj_image(rows(i), cols(i), 3) = curcolor(3);
    end
    bw_image = sum(obj_image,3)>0;
    BBox = regionprops(bw_image, 'BoundingBox' );
    fprintf(BB_file,'%d\n',length(BBox));
    for i_bbox = 1:length(BBox),
        bbox = BBox(i_bbox).BoundingBox;
        fprintf(BB_file,'%d %d %d %d\n',round(bbox(1)),round(bbox(2)),round(bbox(3)),round(bbox(4)));
    end
    gt_image = gt_image + obj_image;
end

if b_display == true,
    subplot(1, 3, 2); 
    imshow(gt_image); hold on; title('Ground Truth'); 

    % % 3. visualize legend
    subplot(1, 3, 3); 
    axis off; hold on; % show off the axis 
    for i=1:length(cur_labels)
        [rows cols] = find(groundtruth == cur_labels(i));
        plot(cols, rows, 's', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', colors(cur_labels(i)+1, :), 'MarkerSize', 10, 'visible', 'off');
    end
    set(gca, 'Ydir', 'reverse'); hold off;
    legend(label_names, 'Location', 'West');
    %pause; 
    close all;
end
end




