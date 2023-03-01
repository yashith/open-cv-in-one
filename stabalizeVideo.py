from vidstab import VidStab
import matplotlib.pyplot as plt

stabilizer = VidStab()
stabilizer.stabilize(input_path='V_392.mp4', output_path='stable_video.avi')

stabilizer.plot_trajectory()
plt.show()

stabilizer.plot_transforms()
plt.show()