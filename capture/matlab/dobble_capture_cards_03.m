
global db;
global im_raw;
global im_pre;
global nCard;
%global S;

N = 57; % total number of cards
R = [120 150]; % range of radius to find
b = 10; % borders to keep
%S = [224 224]; % dimensions of resized image

db = 'dobble_cards_capture_01';
mkdir(db);

%camera = webcam;
%camera = webcam('HD Pro Webcam C920');
camera = webcam(2);

f = figure;
%set(f,'ButtonDownFcn',@(~,~)disp('figure'),'HitTest','off');
set(f,'ButtonDownFcn',@capture_card,'HitTest','off');

nCard = 1;
while true
    im_raw = camera.snapshot;
    imshow(im_raw);
    hold on;

    szTitle = sprintf('Dobble - Capturing Card %d',nCard);
    title(szTitle);

    [centers,radii] = imfindcircles(im_raw,R,'ObjectPolarity','bright');
    h1 = viscircles(centers,radii);
    
    %for c = 1:length(radii)
    if ( length(radii) > 0 )
        c = 1;
        r = uint32(radii(c));
        xc = uint32(centers(c,1));
        yc = uint32(centers(c,2));
        xl = xc - r - b;
        yl = yc - r - b;
        xr = xc + r + b;
        yr = yc + r + b;
        xmax = size(im_raw,2);
        ymax = size(im_raw,1);
        %disp([xl,xr,xmax,yl,yr,ymax]);
        if ( (xl>0) & (yl>0) & (xr<xmax) & (yr<ymax) )
           mask = false(size(im_raw,1:2));
           mask(yl:yr,xl:xr) = true;
           h2 = visboundaries(mask,'Color','b');
           im_pre = im_raw(yl:yr,xl:xr,:);
        end
    end
    
    drawnow;
    
    if ( nCard > N )
        disp('Done!\n');
        exit;
    end
end


function capture_card(hObject,~)
    global db;
    global im_raw;
    global im_pre;
    global nCard;
    %global S;

    dd = fullfile( db, sprintf('%02d',nCard) );
    mkdir(dd);
    df = fullfile( dd, sprintf('card%02d_01.tif',nCard) );
    %im = imresize(im_pre,S);
    im = im_pre;
    imwrite(im,df);

    szMessage = sprintf('Card %d Captured !',nCard);
    disp(szMessage);

    nCard = nCard + 1;
end

