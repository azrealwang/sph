% A demo script showing how to use Gabor features, KFA and a phototmetric normalization technique from the INFace toolbox for face recognition
% 
% DEMO NAME: INFACE GABOR KFA FACE RECOGNITION AND EVALUATION DEMO
% 
% GENERAL DESCRIPTION
% The script demonstrates how to use Gabor magnitude features, KFA and a 
% phototmetric normalization technique from the INFace toolbox for 
% illumination invariant face recognition on 
% real image data and how to classify face images based on Gabor magnitude 
% features and KFA and the nearest neighbor classifier. For matching score 
% calculation the 'mahcos' distance is used. After computing the similarity
% matrix the results are evaluated and some graphical and numerical results 
% are shown. For photometric normalization we use the Tan and Triggs 
% normalization scheme. 
% 
% Note that this demo uses a different partitioning of the input data when
% compared to the KFA_face_recognition_and_evaluation_demo. Here, we
% partition the data into three sets, where the first represents the 
% training data, the second represents the evaluation data (where hyper 
% parameters such as decision threshold are set) and the third set 
% represents the actual test data. This setting makes it possible to create 
% EPC cruves that were not created in the KFA_face_recognition_and_evaluation_demo.
% As a consequence you cannot compare the results directly, but have to
% implement the same partitioning scheme for the GAB+KFA features yourself.
% 
% The script shows how to use the functions from the toolbox to create an
% almost complete face recognition system (the face detection and 
% preprocessing steps are missing) and how to generate performance
% metrics.
% 
% For a more detailed description of the demo please have a look at the user
% manual.
% 
% Abbriviation of the system: INFACE+GAB+KFA+MAHCOS
% 
% The demo assumes that you have downloaded the ORL database and have
% unpacked it to the /demos/database folder. This folder should now have
% the following internal structure:
% 
%   /demos/database/ --- s1/
%                    |-- s2/
%                    |-- s3/
%                    |-- s4/
%                    |-- s5/
%                    |-- s6/
%                    ...
%                    |-- s40/
% 
% Each of the 40 subfolders should contain 10 images in PGM format. If you
% have not downloaded and unpacked the ORL database at all or have unpacked
% it into a different folder this demo will not work!!!
% 
% Please follow the install instructions in the install script.
% 
% IMPORTANT!!!!
% Note that you must run all demo scipts in this toolbox from the demos
% folder. This is particularly important, since some data needed by the
% scripts is located in folders whose paths are specified relative to the 
% demos folder. If you run the scripts from anywhere else, the scripts may
% fail.
% 
% 
% NOTES / COMMENTS
% The script was tested with Matlab version 7.11.0.584 (R2010b) running 
% on a 64-bit Windows 7 OS.
%
% 
% ABOUT
% Created:        30.11.2011
% Last Update:    21.12.2011
% Revision:       1.0
% 
%
% WHEN PUBLISHING A PAPER AS A RESULT OF RESEARCH CONDUCTED BY USING THIS CODE
% OR ANY PART OF IT, MAKE A REFERENCE TO THE FOLLOWING PUBLICATIONS:
% 
% �truc V., Pave�ic, N.: The Complete Gabor-Fisher Classifier for Robust 
% Face Recognition, EURASIP Advances in Signal Processing, vol. 2010, 26
% pages, doi:10.1155/2010/847680, 2010.
%
% �truc V., Pave�ic, N.:Gabor-Based Kernel Partial-Least-Squares 
% Discrimination Features for Face Recognition, Informatica (Vilnius), vol.
% 20, no. 1, pp. 115-138, 2009.
% 
% 
% The BibTex entries for the papers are here
% 
% @Article{ACKNOWL1,
%     author = "Vitomir \v{S}truc and Nikola Pave\v{s}i\'{c}",
%     title  = "The Complete Gabor-Fisher Classifier for Robust Face Recognition",
%     journal = "EURASIP Advances in Signal Processing",
%     volume = "2010",
%     pages = "26",
%     year = "2010",
% }
% 
% @Article{ACKNOWL2,
%     author = "Vitomir \v{S}truc and Nikola Pave\v{s}i\'{c}",
%     title  = "Gabor-Based Kernel Partial-Least-Squares Discrimination Features for Face Recognition",
%     journal = "Informatica (Vilnius)",
%     volume = "20",
%     number = "1",
%     pages = "115�138",
%     year = "2009",
% }
% 
% Official website:
% If you have down-loaded the toolbox from any other location than the
% official website, plese check the following link to make sure that you
% have the most recent version:
% 
% http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/PhDface/index.html
%
% 
% OTHER TOOLBOXES 
% If you are interested in face recognition you are invited too have a look
% at the INface toolbox as well. It contains implementations of several
% state-of-the-art photometric normalization techniques that can further 
% improve the face recognition performance, especcially in difficult 
% illumination conditions. The toolbox is available from:
% 
% http://luks.fe.uni-lj.si/sl/osebje/vitomir/face_tools/INFace/index.html
% 
%
% Copyright (c) 2011 Vitomir �truc
% Faculty of Electrical Engineering,
% University of Ljubljana, Slovenia
% http://luks.fe.uni-lj.si/en/staff/vitomir/index.html
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files, to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in 
% all copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.
% 
% November 2011



%% Load sample image
disp(sprintf('This is a demo script for the PhD toolbox. It demonstrates how to use several \nfunctions from the toolbox to construct a KFA-based face recognition system \nusing Gabor magnitude features and how to test it on the ORL database. At the end, some results \nare generated. Note that the data is partitioned into three sets: \n training, evaluation and test sets. Note that you require the INFace toolbox for this demo or it will fail.'));
disp(' ')
disp(sprintf('Since some of the steps in this demo take a while to complete it may be a good \nidea to turn on reporting.'))
reply = input('Would you like me to turn on reporting? Y/N [Y]: ', 's');
verbose = 1;
if(strcmp(reply,'N') || strcmp(reply,'n'))
    verbose = 0;
end
if(verbose == 0)
    disp('Reporting turned off.');
else
    disp('Reporting turned on.');
end
disp(' ')
disp('Step 1:')
disp(sprintf('Load images from a database, do preprocessing and compute Gabor magnitude features. In our case \nthe images are loaded from the ORL database. During this step we also need to \nconstruct the Gabor filter bank and extract the Gabor magnitude features. \This may take a while.'))

% construct Gabor filter bank 
filter_bank = construct_Gabor_filters_PhD(8, 5, [128 128]);

%filter image
proceed = 1;
data_matrix = [];
ids = [];
conut = 0;
try
    % construct image string, load image and extract features
    for i=1:40
        for j=1:10
            s = sprintf('database/s%i/%i.pgm',i,j);
            X = double(imread(s));
            X = imresize(X,[128 128],'bilinear');
            X = tantriggs(X);
            feature_vector = filter_image_with_Gabor_bank_PhD(X,filter_bank,64);
            data_matrix = [data_matrix,feature_vector];
            ids = [ids;i];
            conut = conut+1;
            if(verbose)
                disp(sprintf('Finished with feature extraction from image %i/%i',conut,400));
            end
        end
    end
    [size_y,size_x] = size(X);    
catch
   proceed = 0;
   disp(sprintf('Could not load images from the ORL database. Did you unpack it into \nthe appropriate directory? If NOT please follow the instructions \nin the user manual or the provided install script. Ending demo prematurely.'));
end






if(proceed)
    disp('Finished with Step 1 (database loading).')
    disp('Press any key to continue ...')
    pause();

    %% Partitioning of the data
    disp(' ')
    disp('Step 2:')
    disp('Partition data into training, evaluation and test sets. In our case, the first 3 images')
    disp('of each ORL subject will serve as the training/gallery/target set, the next three images will')
    disp('serve as the evaluation set and the remaining images will serve as test image set.')

    ids_train = [];
    ids_test = [];
    ids_eval = [];
    train_data = [];
    test_data = [];
    eval_data = [];
    cont = 1;
    for i=1:40
        for j=1:10
            if j<4
                train_data = [train_data,data_matrix(:,cont)];
                ids_train  = [ids_train, ids(cont)];
            elseif j>=4 && j<7
                eval_data = [eval_data,data_matrix(:,cont)];
                ids_eval  = [ids_eval, ids(cont)];
            else
                test_data = [test_data,data_matrix(:,cont)];
                ids_test  = [ids_test,ids(cont)];
            end
            cont = cont + 1;
            if(verbose)
                disp(sprintf('Finished assigning image %i/%i to appropriate image set',cont-1,400));
            end
        end 
    end
    disp('Finished with Step 2 (data partitioning).')
    disp('Press any key to continue ...')
    pause();


	%% Construct KFA subspace
    disp(' ')
    disp('Step 3:')
    disp('Compute training, evaluation and test feature vectors using your method of choice. In our')
    disp('case we use KFA for dimensionality reduction, and, therefore, first compute the KFA ')
    disp('subspace using the training data from the ORL database.')
    model = perform_kfa_PhD(train_data, ids_train, 'fpp', [0 0.7],length(unique(ids_train))-1);
    
    
    disp('Finished KFA subspace construction. Starting evaluation and test image projection.')
    test_features = nonlinear_subspace_projection_PhD(test_data, model);
    eval_features = nonlinear_subspace_projection_PhD(eval_data, model);
    disp('Finished with Step 3 (feature extraction).')
    disp('Press any key to continue ...')
    pause();
    
    
    
    %% Compute similarity matrix
    disp(' ')
    disp('Step 4:')
    disp('Compute matching scores between gallery/training/target feature vectors and')
    disp('evaluation feature vectors. In our case we use the Mahalanobis cosine similarity')
    disp('measure for that.')
    results = nn_classification_PhD(model.train, ids_train, eval_features, ids_eval, size(eval_features,1), 'mahcos');
    disp('We do the same for the test features.')
    results1 = nn_classification_PhD(model.train, ids_train, test_features, ids_test, size(test_features,1), 'mahcos');
    disp('Finished with Step 4 (matching).')
    disp('Press any key to continue ...')
    pause();
    
    
    %% Evaluate similarity matrix
    disp(' ')
    disp('Step 5:')
    disp('Evaluate results and present performance metrics.')
    output = evaluate_results_PhD(results,'ID',results1);
    figure(1)
    plot_ROC_PhD(output.ROC_ver_rate, output.ROC_miss_rate,'r',2);
    title('ROC curve for the INFACE+GAB+KFA+MAHCOS technique on the ORL database.')
    legend('INFACE+GAB+KFA+MAHCOS')
    figure(2)
    plot_CMC_PhD(output.CMC_rec_rates , output.CMC_ranks,'r',2);
    legend('INFACE+GAB+KFA+MAHCOS')
    title('CMC curve for the INFACE+GAB+KFA+MAHCOS technique on the ORL database.')
    try
        figure(3)    
        Plot_DET(output.DET_frr_rate, output.DET_far_rate, 'r', 2);
        title('DET curve for the INFACE+GAB+KFA+MAHCOS technique on the ORL database.')
    catch
        close 3    
        disp('Tried to plot DET curve, but it seems you have not installed NISTs DETware.')
    end
    figure(4)
    plot_EPC_PhD(output.EPC_alpha,output.EPC_errors,'r',2);
    legend('INFACE+GAB+KFA+MAHCOS')
    title('EPC curve for theINFACE+ GAB+KFA+MAHCOS technique on the ORL database.')
    disp(' ')
    disp('=============================================================')
    disp('SOME PERFORMANCE METRICS:')
    disp(' ')
    disp('Identification experiments:')
    disp(sprintf('The rank one recognition rate of the experiments equals (in %%): %3.2f%%', output.CMC_rec_rates(1)*100));
    disp(' ')
    disp('Verification/authentication experiments on the evaluation data:')
    disp(sprintf('The equal error rate on the evaluation set equals (in %%): %3.2f%%', output.ROC_char_errors.EER_er*100));
    disp(sprintf('The minimal half total error rate on the evaluation set equals (in %%): %3.2f%%', output.ROC_char_errors.minHTER_er*100));
    disp(sprintf('The verification rate at 1%% FAR on the evaluation set equals (in %%): %3.2f%%', output.ROC_char_errors.VER_1FAR_ver*100));
    disp(sprintf('The verification rate at 0.1%% FAR on the evaluation set equals (in %%): %3.2f%%', output.ROC_char_errors.VER_01FAR_ver*100));
    disp(sprintf('The verification rate at 0.01%% FAR on the evaluation set equals (in %%): %3.2f%%', output.ROC_char_errors.VER_001FAR_ver*100));
    disp(' ')
    disp('Verification/authentication experiments on the test data (preset thresholds on evaluation data):')
    disp(sprintf('The verification rate at 1%% FAR on the test set equals (in %%): %3.2f%%', output.EPC_char_errors.test_VER_1FAR_ver*100));
    disp(sprintf('The verification rate at 0.1%% FAR on the test set equals (in %%): %3.2f%%', output.EPC_char_errors.test_VER_01FAR_ver*100));
    disp('=============================================================')

    disp('Finished with Step 5 (evaluation).')
    
end
disp('Finished demo.')




















