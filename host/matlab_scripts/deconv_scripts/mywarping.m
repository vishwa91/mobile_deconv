function [warped]=mywarping(img,homo)
[Y,X]=size(img);
img_out=zeros(size(img));
for y=1:Y
    for x=1:X         
        dummy= homo*[x y 1]';
        posx=dummy(1)/dummy(3);
        posy=dummy(2)/dummy(3);
        dx=posx-floor(posx);
        dy=posy-floor(posy);
        if(floor(posx)>0 && floor(posy)>0 && ceil(posx)<=X && ceil(posy)<=Y)
        img_out(y,x)=(1-dy)*(1-dx)*img(floor(posy),floor(posx)) ...
                     +dy*(1-dx)*img(ceil(posy),floor(posx)) ...
                     +(1-dy)*dx*img(floor(posy),ceil(posx)) ...
                     +dy*dx*img(ceil(posy),ceil(posx));
        end
    end
end
warped=img_out;