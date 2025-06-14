%% Implementare CNN
clc;
clear;

%% 1. Setări directoare
trainFolder = 'train';
validFolder = 'valid';
testFolder  = 'test';

%% 2. Citim datele ca imageDatastore
imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsValid = imageDatastore(validFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest  = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% 3. Redimensionare imagini + conversie grayscale
inputSize = [64 64];  % dimensiune imagine de intrare
augTrain = augmentedImageDatastore([inputSize 1], imdsTrain); % grayscale
augValid = augmentedImageDatastore([inputSize 1], imdsValid);
augTest  = augmentedImageDatastore([inputSize 1], imdsTest);

%% 4. Arhitectură CNN simplă
layers = [
    imageInputLayer([inputSize 1])  % grayscale

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(unique(imdsTrain.Labels)))
    softmaxLayer
    classificationLayer
];

%% 5. Setări pentru antrenare
options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augValid, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

%% 6. Antrenare rețea 
% net = trainNetwork(augTrain, layers, options);
% save('CNNmodel.mat', 'net');

load("CNNmodel.mat");  % încarcă rețeaua antrenată

%% 7. Clasificare pe tot setul de test
YPred = classify(net, augTest);
YTrue = imdsTest.Labels;

% Acuratețe generală
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Acuratețea pe setul de test: %.2f%%\n', accuracy * 100);

% Matrice de confuzie
figure;
confusionchart(YTrue, YPred);
title('Matrice de Confuzie - Set Test');


%% 8. Afișare aleatorie din fiecare clasă cu comparație etichete

uniqueLabels = unique(imdsTest.Labels);
inputSize = [64 64];  % după cum e definit mai sus

for i = 1:numel(uniqueLabels)
    currentLabel = uniqueLabels(i);

    % Găsește toate imaginile din clasa curentă
    idx = find(imdsTest.Labels == currentLabel);

    if isempty(idx)
        continue;
    end

    % Selectează una aleatoare
    randIdx = idx(randi(numel(idx)));
    I = readimage(imdsTest, randIdx);
    trueLabel = imdsTest.Labels(randIdx);

    % Resize și conversie la RGB dacă e nevoie
    I_input = imresize(I, inputSize);
    if size(I_input, 3) == 1
        I_input = cat(3, I_input, I_input, I_input);
    end

    % Clasificare
    predictedLabel = classify(net, I_input);

    % Afișare
    figure;
    imshow(I, []);
    
    % Adăugarea unei casete de text pentru eticheta reală și prezisă
    trueLabelText = sprintf('Etichetă reală: %s', string(trueLabel));
    predictedLabelText = sprintf('Etichetă prezisă: %s', string(predictedLabel));

    % Dimensiuni și locație pentru caseta de text
    dim = [0.1 0.8 0.8 0.15];  % dimensiunea casetei de text (x, y, lățime, înălțime)
    str = sprintf('%s\n%s', trueLabelText, predictedLabelText);  % combină etichetele într-un singur text

    % Afișează caseta de text cu fundal
    annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
        'BackgroundColor', 'black', 'Color', 'white', 'FontSize', 14, 'FontWeight', 'bold');

    pause();  % scurtă pauză pentru a vizualiza
end
%% Clasificare per clasa

% Etichete unice din setul de test
classes = unique(YTrue);

fprintf('--- Clasificare corectă per clasă ---\n');
for i = 1:numel(classes)
    cls = classes(i);

    % Total imagini reale din această clasă
    totalReal = sum(YTrue == cls);

    % Câte au fost prezise corect
    corecte = sum((YTrue == cls) & (YPred == cls));

    fprintf('%s: %d din %d corecte (%.2f%%)\n', ...
        string(cls), corecte, totalReal, 100 * corecte / totalReal);
end
