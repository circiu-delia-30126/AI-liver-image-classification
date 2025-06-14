%% Implementare Random Forest 
clc;
clear;

%% 1. Setări directoare 
trainFolder = 'train';
validFolder = 'valid';
testFolder  = 'test';

%% 2. Citim datele
imdsTrain = imageDatastore(trainFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsValid = imageDatastore(validFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imdsTest  = imageDatastore(testFolder, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Funcție pentru extragerea caracteristicilor din imagini
function features = extractRandomForestFeatures(img, inputSize)
    % Această funcție extrage caracteristici relevante pentru Random Forest
    
    % 1. Asigură-te că imaginea este la dimensiunea corectă
    img = imresize(img, inputSize);
    
    % 2. Extrage caracteristici HOG (Histogram of Oriented Gradients)
    [hogFeatures, ~] = extractHOGFeatures(img);
    
    % 3. Extrage caracteristici GLCM (Gray-Level Co-Occurrence Matrix)
    glcm = graycomatrix(img, 'Offset', [0 1; -1 1; -1 0; -1 -1]);
    statsGLCM = graycoprops(glcm, {'contrast', 'correlation', 'energy', 'homogeneity'});
    
    % Transformă structura GLCM într-un vector
    glcmFeatures = [statsGLCM.Contrast(:); statsGLCM.Correlation(:); ...
                    statsGLCM.Energy(:); statsGLCM.Homogeneity(:)];
    
    % 4. Histograma de intensități
    histFeatures = imhist(img, 32);
    histFeatures = histFeatures / sum(histFeatures); % Normalizare
    
    % 5. Extrage LBP (Local Binary Pattern)
    lbpFeatures = extractLBPFeatures(img, 'CellSize', [16 16], 'Normalization', 'none');
    
    % 6. Combină toate caracteristicile
    % Poți ajusta dimensiunile pentru a te asigura că obții 512 caracteristici
    features = [hogFeatures(:)', glcmFeatures', histFeatures', lbpFeatures(:)'];
    
    % Asigură-te că are exact 512 caracteristici (sau ajustează în funcție de necesități)
    if length(features) > 512
        features = features(1:512);
    elseif length(features) < 512
        % Completează cu zerouri dacă avem mai puține caracteristici
        features = [features, zeros(1, 512 - length(features))];
    end
end

%% 3. Redimensionare imagini + extragere caracteristici
inputSize = [64 64];  % dimensiune imagine de intrare

% Funcție pentru extragerea caracteristicilor din imagini
extractFeatures = @(img) extractRandomForestFeatures(img, inputSize);

% Extragere caracteristici pentru seturile de date
featuresTrain = zeros(numel(imdsTrain.Files), 512);  % alocăm spațiu pentru features
labelsTrain = imdsTrain.Labels;

% Extragere caracteristici din setul de antrenare
for i = 1:numel(imdsTrain.Files)
    img = readimage(imdsTrain, i);
    img = imresize(img, inputSize);
    if size(img, 3) == 3
        img = rgb2gray(img); % Conversie la grayscale dacă este color
    end
    featuresTrain(i, :) = extractFeatures(img);
    
    % Afișare progres
    if mod(i, 100) == 0
        fprintf('Procesare imagini antrenare: %d/%d\n', i, numel(imdsTrain.Files));
    end
end

% Extragere caracteristici din setul de validare
featuresValid = zeros(numel(imdsValid.Files), 512);
labelsValid = imdsValid.Labels;

for i = 1:numel(imdsValid.Files)
    img = readimage(imdsValid, i);
    img = imresize(img, inputSize);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    featuresValid(i, :) = extractFeatures(img);
    
    if mod(i, 100) == 0
        fprintf('Procesare imagini validare: %d/%d\n', i, numel(imdsValid.Files));
    end
end

% Extragere caracteristici din setul de test
featuresTest = zeros(numel(imdsTest.Files), 512);
labelsTest = imdsTest.Labels;

for i = 1:numel(imdsTest.Files)
    img = readimage(imdsTest, i);
    img = imresize(img, inputSize);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    featuresTest(i, :) = extractFeatures(img);
    
    if mod(i, 100) == 0
        fprintf('Procesare imagini test: %d/%d\n', i, numel(imdsTest.Files));
    end
end

%% 4. Antrenarea modelului Random Forest
fprintf('Antrenare model Random Forest...\n');

% Setare hiperparametri pentru Random Forest
numTrees = 100;  % Număr de arbori
minLeafSize = 5; % Dimensiunea minimă a unei frunze

% Antrenare model
model = TreeBagger(numTrees, featuresTrain, labelsTrain, ...
    'Method', 'classification', ...
    'MinLeafSize', minLeafSize, ...
    'OOBPrediction', 'on', ...
    'OOBPredictorImportance', 'on');

%save('randomForestModel.mat', 'model');
load('randomForestModel.mat', 'model');
%% metode de validarea interna

% Calculare predicții Out-of-Bag (pentru a evalua performanța pe setul de antrenare)
oobError = oobError(model);
fprintf('Eroare OOB finală: %.4f\n', oobError(end));

% Vizualizare importanță predictori
figure;
bar(model.OOBPermutedPredictorDeltaError);
title('Importanța variabilelor în model');
xlabel('Indice predictor');
ylabel('Importanță Out-of-Bag');

%% 6. Evaluare pe setul de test
[predictedLabelsTest, scoresTest] = predict(model, featuresTest);
predictedLabelsTest = categorical(predictedLabelsTest);

% Calculare acuratețe
accuracyTest = sum(predictedLabelsTest == labelsTest) / numel(labelsTest);
fprintf('Acuratețe pe setul de test: %.2f%%\n', accuracyTest * 100);

% Matrice de confuzie pentru test
figure;
confusionchart(labelsTest, predictedLabelsTest);
title('Matrice de Confuzie - Set Test');

%% 7. Afișare exemple clasificate
fprintf('--- Clasificare corectă per clasă ---\n');
uniqueClasses = unique(labelsTest);

for i = 1:numel(uniqueClasses)
    cls = uniqueClasses(i);
    
    % Total imagini reale din această clasă
    totalReal = sum(labelsTest == cls);
    
    % Câte au fost prezise corect
    corecte = sum((labelsTest == cls) & (predictedLabelsTest == cls));
    
    fprintf('%s: %d din %d corecte (%.2f%%)\n', ...
        string(cls), corecte, totalReal, 100 * corecte / totalReal);
end

%% 8. Vizualizare imagini clasificate
for i = 1:numel(uniqueClasses)
    currentClass = uniqueClasses(i);
    
    % Găsirea unei imagini din clasa curentă
    idx = find(labelsTest == currentClass);
    
    if isempty(idx)
        continue;
    end
    
    % Selectează random o imagine
    randIdx = idx(randi(numel(idx)));
    img = readimage(imdsTest, randIdx);
    
    % Citire etichetă reală și prezisă
    trueLabel = labelsTest(randIdx);
    predictedLabel = predictedLabelsTest(randIdx);
    
    % Afișare imagine cu etichetele
    figure;
    imshow(img);
    title(['Real: ' char(trueLabel) ', Prezis: ' char(predictedLabel)]);
end

