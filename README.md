# Pre-trained Integrated Model for Earthquake Data Processing
We introduce a new seismic wave representation model called PRIME-DP, which stands for Pre-trained Integrated Model for Earthquake Data Processing. Unlike most of the models, which are designed to specifically a singular problem, PRIME-DP is used for multi-task single station seismic waveform processing. PRIME-DP can be used to Pg/Sg/Pn/Sn phase picking, P polarization classification. And can be fine-tunned to wide range of tasks, such as event classification, without architecture modifications. PRIME-DP can achieve over 85% recall on Pg and Sg phases, when picking continuous waveform and achieves over 80% accuracy in P polarization classification. By fine-tunning classification decoder with NeiMeng dataset, PRIME-DP can achieve 95.1% accuracy on event.


# Author 
You can contact yuziye@cea-igp.ac.cn for the weight file. 