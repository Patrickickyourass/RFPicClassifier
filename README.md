Quality control for Receiver Functions using Refined AlexNet, SAC files -->Pictures-->Classify

Wiggle_RF and Wiggle in the format of the matlab is to draw the sac files as wiggle pics

RF_classify is to classify the RFs and provide a result

Pre-trained Model Weights:https://github.com/Patrickickyourass/RFPicClassifier/releases/tag/RFPicpicker

Receiver Function MethodologyThe P-wave receiver functions (PRFs) were computed using the iterative time-domain deconvolution technique implemented in the Computer Programs in Seismology (CPS) package. For each earthquake-station pair, the radial component was deconvolved by the vertical component to isolate the P-to-S converted phases and their multiples. Specifically, we applied a Gaussian filter with a factor of $\alpha = 2.0$ to suppress high-frequency noise, which effectively limits the frequency content below approximately $0.64$ Hz. A time delay of $5$ s was introduced to the beginning of the receiver functions to preserve the direct P-wave arrival. The deconvolution process was performed with a maximum of $100$ iterations to ensure a high-quality fit between the observed and predicted waveforms.
