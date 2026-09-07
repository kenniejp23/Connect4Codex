import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    objectName: "trainingPage"
    property bool advanced: false
    property bool resuming: false
    property var fullConfig: ({})
    property string configurationError: ""
    property string submittedConfiguration: ""
    Component.onCompleted: loadRun()

    function showConfig(config) {
        fullConfig = config
        device.currentIndex = Math.max(0, device.find(config.device))
        selfPlayGames.value = config.replay_warmup_games
        trainingMcts.value = config.n_mcts_iterations
        selfPlayBatch.value = config.inference_batch_size
        trainBatch.value = config.training_batch_size
        residualBlocks.value = config.n_residual_blocks
        filters.value = config.conv_filter_size
        policyLayers.value = config.n_policy_layers
        valueLayers.value = config.n_value_layers
        exploration.text = String(config.c_exploration)
        plyPenalty.text = String(config.c_ply_penalty)
        learningSchedule.text = config.lr_schedule.join(", ")
        l2.text = String(config.l2_reg)
        if (config.max_gens !== null) generations.value = config.max_gens
        var immutable = [selfPlayGames, trainingMcts, selfPlayBatch, trainBatch, residualBlocks,
                         filters, policyLayers, valueLayers, exploration, plyPenalty, learningSchedule, l2]
        immutable.forEach(function(control) { control.enabled = !page.resuming })
    }

    function loadRun() {
        var info = JSON.parse(App.trainingConfiguration(trainingDirectory.text))
        configurationError = info.error || ""
        if (info.error) return
        resuming = info.resume
        showConfig(info.config)
    }
    property bool compact: width < 1000

    function scheduleValues() {
        return learningSchedule.text.split(",").map(function(value) { return Number(value.trim()) })
    }

    function submitTraining() {
        var shardGames = Math.min(256, selfPlayGames.value)
        var batchGames = Math.min(
            shardGames * 2,
            Math.floor(selfPlayGames.value / shardGames) * shardGames
        )
        var edits = {
            base_dir: trainingDirectory.text,
            device: device.currentText,
            replay_warmup_games: selfPlayGames.value,
            self_play_shard_games: shardGames,
            self_play_batch_games: batchGames,
            replay_capacity_games: Math.max(20000, selfPlayGames.value + 1),
            replay_ratio: 4.0,
            validation_fraction: 0.05,
            archive_depth: 8,
            n_mcts_iterations: trainingMcts.value,
            c_exploration: Number(exploration.text),
            c_ply_penalty: Number(plyPenalty.text),
            inference_batch_size: selfPlayBatch.value,
            training_batch_size: trainBatch.value,
            n_residual_blocks: residualBlocks.value,
            conv_filter_size: filters.value,
            n_policy_layers: policyLayers.value,
            n_value_layers: valueLayers.value,
            lr_schedule: scheduleValues(),
            l2_reg: Number(l2.text),
            max_gens: generations.value,
            arena_min_games: 40,
            arena_max_games: 800,
            root_dirichlet_epsilon: 0.25,
            root_dirichlet_alpha: 0.30,
            temperature_cutoff_ply: 8
        }
        var config = Object.assign({}, fullConfig, edits)
        if (resuming) config = Object.assign({}, fullConfig, {
            base_dir: trainingDirectory.text, device: device.currentText, max_gens: generations.value
        })
        submittedConfiguration = JSON.stringify(config, null, 2)
        Jobs.submit("training", JSON.stringify(config), "Train " + trainingDirectory.text)
    }

    function applyPreset(name) {
        if (resuming) return
        showConfig(JSON.parse(App.trainingPreset(name)))
    }

    RowLayout {
        anchors.fill: parent
        anchors.leftMargin: 28
        anchors.rightMargin: 28
        anchors.bottomMargin: 24
        spacing: 16

        Panel {
            objectName: "trainingConfigPanel"
            Layout.fillWidth: page.compact
            Layout.preferredWidth: page.compact ? 0 : 680
            Layout.fillHeight: true
            Layout.minimumWidth: page.compact ? 440 : 570
            ScrollView {
                anchors.fill: parent
                anchors.margins: 20
                clip: true
                Column {
                    width: parent.width
                    spacing: 18
                    SectionTitle {
                        width: parent.width
                        title: "Configure training"
                        subtitle: "Run asynchronous neural self-play, replay training, and champion gating."
                    }

                    Label { width: parent.width; wrapMode: Text.Wrap; text: page.configurationError || (page.resuming ? "Resuming saved experiment. Experiment settings are locked. Choose an empty directory to change them. Champion limit is run-wide." : "New experiment"); color: ApplicationWindow.window.mutedTextColor }
                    TextArea { width: parent.width; visible: page.submittedConfiguration.length > 0; text: page.submittedConfiguration; readOnly: true; wrapMode: TextEdit.Wrap; Accessible.name: "Submitted training configuration" }
                    Text { text: "PRESET"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                    Row {
                        spacing: 8
                        Button { enabled: !page.resuming; text: "CPU"; onClicked: page.applyPreset("CPU") }
                        Button { enabled: !page.resuming; text: "Balanced"; onClicked: page.applyPreset("Balanced") }
                        PrimaryButton { enabled: !page.resuming && App.cudaAvailable; text: "GPU"; onClicked: page.applyPreset("GPU") }
                    }

                    GridLayout {
                        width: parent.width
                        columns: 2
                        columnSpacing: 14
                        rowSpacing: 12
                        LabeledField { id: trainingDirectory; Layout.fillWidth: true; label: "Training directory"; text: App.trainingDir; onEditingFinished: { App.setTrainingDir(text); page.loadRun() } }
                        Column {
                            Layout.fillWidth: true
                            spacing: 6
                            Text { text: "Device"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12; font.weight: Font.Medium }
                            ComboBox { id: device; width: parent.width; height: 44; model: App.cudaAvailable ? ["cuda", "cpu"] : ["cpu"]; currentIndex: 0 }
                        }
                        Column {
                            Layout.fillWidth: true; spacing: 6
                            Text { text: "Target accepted champions"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                            SpinBox { id: generations; width: parent.width; height: 44; from: 1; to: 10000; value: 10; editable: true }
                        }
                        Column {
                            Layout.fillWidth: true; spacing: 6
                            Text { text: "Fresh games / candidate"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                            SpinBox { id: selfPlayGames; width: parent.width; height: 44; from: 2; to: 10000000; value: 2048; editable: true }
                        }
                        Column {
                            Layout.fillWidth: true; spacing: 6
                            Text { text: "MCTS iterations / move"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                            SpinBox { id: trainingMcts; width: parent.width; height: 44; from: 1; to: 1000000; value: 1400; editable: true }
                        }
                        LabeledField { id: exploration; Layout.fillWidth: true; label: "Exploration constant"; text: "6.6"; validator: DoubleValidator { bottom: 0 } }
                    }

                    Button {
                        text: page.advanced ? "Hide advanced parameters" : "Show advanced parameters"
                        flat: true
                        onClicked: page.advanced = !page.advanced
                    }

                    Column {
                        visible: page.advanced
                        width: parent.width
                        spacing: 14
                        Text { text: "SELF-PLAY & OPTIMIZATION"; color: ApplicationWindow.window.accentColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                        GridLayout {
                            width: parent.width; columns: 3; columnSpacing: 12; rowSpacing: 12
                            Column { Layout.fillWidth: true
                                Text { text: "Inference batch"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: selfPlayBatch; width: parent.width; from: 1; to: 1000000; value: 128; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Training batch"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: trainBatch; width: parent.width; from: 2; to: 1000000; value: 512; editable: true }
                            }
                            LabeledField { id: plyPenalty; Layout.fillWidth: true; label: "Ply penalty"; text: "0.01"; validator: DoubleValidator { bottom: 0 } }
                            Column { Layout.fillWidth: true
                                Text { text: "Residual blocks"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: residualBlocks; width: parent.width; from: 0; to: 64; value: 1; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Conv filters"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: filters; width: parent.width; from: 1; to: 4096; value: 32; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Policy layers"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: policyLayers; width: parent.width; from: 1; to: 64; value: 4; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Value layers"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: valueLayers; width: parent.width; from: 1; to: 64; value: 2; editable: true }
                            }
                        }
                        LabeledField { id: learningSchedule; width: parent.width; label: "Learning-rate schedule (generation, rate pairs)"; text: "0, 0.002, 10, 0.0008" }
                        LabeledField { id: l2; width: parent.width; label: "L2 regularization"; text: "0.0004"; validator: DoubleValidator { bottom: 0 } }

                    }

                    RowLayout {
                        width: parent.width
                        spacing: 10
                        PrimaryButton { text: Jobs.active ? "Queue training" : "Start training"; onClicked: page.submitTraining() }
                        Text { Layout.fillWidth: true; wrapMode: Text.WordWrap; text: Jobs.active ? "This run will start after " + Jobs.currentTitle : "Replay shards and candidates are saved atomically"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                    }
                }
            }
        }

        Panel {
            objectName: "trainingMonitorPanel"
            Layout.fillWidth: !page.compact
            Layout.preferredWidth: page.compact ? 280 : 440
            Layout.fillHeight: true
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 18
                spacing: 12
                Text { text: "TRAINING MONITOR"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                Text { text: Jobs.active ? Jobs.currentTitle : "No active job"; color: ApplicationWindow.window.textColor; font.pixelSize: 18; font.weight: Font.DemiBold; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                Text { text: Jobs.phase; color: Jobs.active ? ApplicationWindow.window.successColor : ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                GridLayout {
                    Layout.fillWidth: true
                    columns: 2
                    columnSpacing: 8
                    rowSpacing: 8
                    Repeater {
                        model: [
                            { label: "ACTOR", value: Jobs.liveTraining.actor },
                            { label: "TRAINER", value: Jobs.liveTraining.trainer },
                            { label: "ARENA", value: Jobs.liveTraining.arena },
                            { label: "REPLAY", value: Jobs.liveTraining.replay },
                            { label: "CHAMPION", value: Jobs.liveTraining.champion },
                            { label: "CANDIDATE", value: Jobs.liveTraining.candidate }
                        ]
                        Rectangle {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 52
                            radius: 8
                            color: ApplicationWindow.window.inputColor
                            Column {
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 3
                                Text { text: modelData.label; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9; font.weight: Font.DemiBold }
                                Text { width: parent.width; text: modelData.value; color: ApplicationWindow.window.textColor; font.pixelSize: 11; elide: Text.ElideRight }
                            }
                        }
                    }
                }
                ProgressBar { Layout.fillWidth: true; from: 0; to: 1; value: Jobs.progress; indeterminate: Jobs.active && Jobs.progress <= 0 }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Button {
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        Layout.preferredHeight: 40
                        text: Jobs.stopAfterGenerationRequested ? "Stop scheduled" : (page.compact ? "Stop" : "Stop after candidate")
                        visible: Jobs.active && Jobs.currentKind === "training"
                        enabled: !Jobs.stopAfterGenerationRequested
                        onClicked: Jobs.stopAfterGeneration()
                    }
                    Button {
                        Layout.preferredWidth: page.compact ? 88 : 92
                        Layout.minimumWidth: page.compact ? 88 : 92
                        Layout.preferredHeight: 40
                        leftPadding: 10
                        rightPadding: 10
                        text: "Cancel"
                        visible: Jobs.active
                        onClicked: Jobs.cancel()
                    }
                }
                Button { text: "Clear log"; flat: true; Layout.alignment: Qt.AlignRight; onClicked: Jobs.clearLogs() }
                Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 1; color: ApplicationWindow.window.borderColor }
                ScrollView {
                    objectName: "trainingLogScroll"
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    ScrollBar.horizontal.policy: ScrollBar.AsNeeded
                    ScrollBar.vertical.policy: ScrollBar.AsNeeded
                    ListView {
                        id: trainingLog
                        objectName: "trainingLog"
                        clip: true
                        model: Jobs.logModel
                        onCountChanged: if (count > 0) positionViewAtEnd()
                        delegate: TextEdit {
                            required property string line
                            width: trainingLog.width
                            text: line
                            color: ApplicationWindow.window.mutedTextColor
                            font.family: "monospace"
                            font.pixelSize: 11
                            readOnly: true
                            selectByMouse: true
                            wrapMode: TextEdit.WrapAnywhere
                        }
                        Label {
                            width: parent.width
                            visible: trainingLog.count === 0
                            text: "Live phases, metrics, and worker output will appear here."
                            wrapMode: Text.Wrap
                            color: ApplicationWindow.window.mutedTextColor
                        }
                    }
                }
                Text { visible: Jobs.result.length > 0; text: Jobs.result; color: ApplicationWindow.window.textColor; font.family: "monospace"; font.pixelSize: 10; wrapMode: Text.WrapAnywhere; Layout.fillWidth: true }
            }
        }
    }
}
