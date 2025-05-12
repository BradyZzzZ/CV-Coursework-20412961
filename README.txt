1. Program Running Steps:

	Step 1: Navigate to the folder where the python file is located

	Step 2: In Terminal, run the entire program by using the command "streamlit run Coursework_20412961.py"

	Step 3: Upload the single Input Video

	Step 4: Manually select the hyperparameter values for detection and tracking

	Step 5: Manually select the body boundary visualization method

	Step 6: Click the "Start to Process" button

	Step 7: Wait until the program running finishes

	Step 8: View the Output Video

	Step 9: Decide whether to download the converted Output Video (using the "Download H.264 Output" button)

	Step 10: Check the output metrics tables and conduct evaluation and analysis


2. Several important Python packages and methods used:

	a. OpenCV (cv2) -- Used for video reading and writing, image pre-processing and drawing visualizations.

		Related used methods: "cv2.VideoCapture", "cv2.cvtColor", "cv2.rectangle", "cv2.putText", "cv2.Canny"

	b. PyTorch -- Used to load and run the YOLOv5-Medium model, perform inference, and manage GPU memory

		Related used methods: "torch.hub.load", "torch.device", "torch.cuda.empty_cache()"

	c. Streamlit -- Provides a simple UI for users to upload videos, adjust parameters and display output videos and metrics tables.

		Related used methods: "st.slider", "st.selectbox", "st.file_uploader"

	d. DeepSORT (deep_sort_realtime) -- Enables person identity tracking across frames

		Related used methods: "DeepSort.update_tracks"

	e. NumPy --  Assists in numerical calculations and trajectory distance computation

		Related used methods: "np.linalg.norm", "np.mean", "np.std"

	f. Pandas -- Used to format and display metrics tables in the web interface

		Related used methods: "pd.DataFrame", "st.dataframe"

	g. Subprocess -- Invokes FFmpeg for video format conversion

		Related used methods: "subprocess.run"

	h. Other basic libraries -- OS, SYS, Time, Types, Wearnings


3. My Python Version is "Python 3.10.13" configured through Anaconda Environment

	For specific versions of each used package, please refer to the "requirements.txt" file

	You can install them automatically through "pip install -r requirements.txt" command

Note: If you have the latest version of the TensorFlow-GPU (even though it will not used) package in your current environment, it is recommended to reconfigure a new Anaconda environment to avoid conflicts.