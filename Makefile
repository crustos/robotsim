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

## Offline gate: needs no Blender, so it is plain python3 rather than headless.py.
test_armulator_bridge:
	cd tests && python3 ./armulator_bridge_test.py

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

test_perception:
	cd tests && ./perception_test.py

## Runs under plain python3, not Blender: the MuJoCo backend is free of bpy on
## purpose, so its physics can be checked without launching a scene.
test_muble:
	cd tests && python3 ./muble_test.py

## Scene descriptions and the multi-task perception head. Plain python3: the
## caption generator is bpy-free apart from one lazy helper.
test_captions:
	cd tests && python3 ./captions_test.py

## Stage 2: the expert, the observation encoding and the control policy.
test_control_policy:
	cd tests && python3 ./control_test.py

## The half that does need Blender: appending MuBlE's .blend assets, applying
## its materials and reading a render back.
test_muble_blender:
	blender --background --python tests/muble_blender_test.py

dataset:
	cd tools && ./generate_dataset.py -- --samples 64 --out /tmp/corpus

## A corpus whose geometry is MuBlE's tabletop scenes rather than robotsim's
## procedural obstacles. MUBLE points at a checkout; export the scenes first:
##   $(MUBLE)/robotsim_export.py <scenes.json> -o /tmp/muble_handoff/
MUBLE ?= ../MuBlE
muble_corpus:
	cd tools && ./generate_dataset.py -- --samples 64 --out /tmp/muble_corpus \
		--muble-scenes /tmp/muble_handoff --muble-root $(abspath $(MUBLE))

## Does the control policy still want more data? Trains on increasing
## fractions of the episodes against a fixed held-out set.
scale_control:
	python3 tools/scale_control.py --corpus /tmp/control

## Stage 2 end to end: expert rollouts -> behaviour cloning -> closed loop.
control_corpus:
	cd tools && ./generate_control.py -- --episodes 30 --steps 14 --out /tmp/control

control_policy:
	python3 tools/train_control.py --corpus /tmp/control --epochs 35

control_eval:
	cd tools && ./evaluate_control.py -- --episodes 8 --policy /tmp/control_net.npz

muble_handoff:
	$(MUBLE)/robotsim_export.py \
		$(MUBLE)/demo_output/scene_generaion/NS_AP_scenes.json \
		-o /tmp/muble_handoff/

corpus:
	./tools/generate_corpus.py --samples 512 --out /tmp/corpus --prune

train:
	./tools/train_perception.py --corpus /tmp/corpus --epochs 40

test_all: test test_anim test_joints test_drive test_record test_arm_record test_rig test_sensors test_lidar test_contact test_firmware test_fleet test_telemetry test_dataset test_perception test_muble test_muble_blender test_captions test_control_policy test_armulator_bridge

install:
	chmod +x robotsim.py
	chmod +x headless.py
	chmod +x tests/*.py
	chmod +x tools/*.py
	sudo apt-get install blender
