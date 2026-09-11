default:
	./robotsim.py

headless:
	./headless.py

test:
	cd tests && ./render_test.py

test_anim:
	cd tests && ./render_anim_test.py

test_joints:
	cd tests && ./joint_control_test.py

test_drive:
	cd tests && ./drive_test.py

test_record:
	cd tests && ./record_test.py

test_arm_record:
	cd tests && ./arm_record_test.py

test_rig:
	cd tests && ./rig_test.py

test_sensors:
	cd tests && ./sensor_test.py

test_lidar:
	cd tests && ./lidar_test.py

test_contact:
	cd tests && ./contact_test.py

test_firmware:
	cd tests && ./firmware_test.py

test_fleet:
	cd tests && ./fleet_test.py

test_telemetry:
	cd tests && ./telemetry_test.py

test_dataset:
	cd tests && ./dataset_test.py

dataset:
	cd tools && ./generate_dataset.py -- --samples 64 --out /tmp/corpus

test_all: test test_anim test_joints test_drive test_record test_arm_record test_rig test_sensors test_lidar test_contact test_firmware test_fleet test_telemetry test_dataset

install:
	chmod +x robotsim.py
	chmod +x headless.py
	chmod +x tests/*.py
	sudo apt-get install blender
