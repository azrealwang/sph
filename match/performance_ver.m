function [EER] = performance_ver(new_featuresA,new_featuresB,expression_set,offset,default_set,dist_type)
% cross modality
lg = size(new_featuresA,1)/default_set;
combination = nchoosek(1:lg, 2);
for expression_gallery=expression_set
    for expression_probe = expression_set
        gen_expression=[];
        imp_expression=[];
        for i=1:lg
            A= new_featuresA((i-1)*default_set+expression_gallery,:);
            B= new_featuresB((i-1)*default_set+expression_probe,:);
            if dist_type == "cosine"
                similarity=1-pdist2(A,B,dist_type)/2;
            else
                similarity=1-pdist2(A,B,dist_type);
            end
            gen_expression=[gen_expression similarity];
        end
        for i = 1:length(combination)
            subject = combination(i,:);
            A= new_featuresA((subject(1)-1)*default_set+expression_gallery,:);
            B= new_featuresB((subject(2)-1)*default_set+expression_probe,:);
            if dist_type == "cosine"
                similarity=1-pdist2(A,B,dist_type)/2;
            else
                similarity=1-pdist2(A,B,dist_type);
            end
            imp_expression=[imp_expression similarity];
        end
        if expression_gallery>offset
            i_gallery=expression_gallery-offset;
        else
            i_gallery=expression_gallery;
        end
        if expression_probe>offset
            i_probe=expression_probe-offset;
        else
            i_probe=expression_probe;
        end
        [EER(i_gallery,i_probe), mTSR, mFAR, mFRR, mGAR] = computeperformance(gen_expression, imp_expression, 0.0001);
    end
end