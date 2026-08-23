import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    objectName: "trainingPage"
    property bool advanced: false
    property bool compact: width < 1000

    function scheduleValues() {
        return learningSchedule.text.split(",").map(function(value) { return Number(value.trim()) })
    }

    function submitTraining() {
        var config = {
            base_dir: trainingDirectory.text,
            device: device.currentText,
            n_self_play_games: selfPlayGames.value,
            n_mcts_iterations: trainingMcts.value,
            c_exploration: Number(exploration.text),
            c_ply_penalty: Number(plyPenalty.text),
            self_play_batch_size: selfPlayBatch.value,
            training_batch_size: trainBatch.value,
            n_residual_blocks: residualBlocks.value,
            conv_filter_size: filters.value,
            n_policy_layers: policyLayers.value,
            n_value_layers: valueLayers.value,
            lr_schedule: scheduleValues(),
            l2_reg: Number(l2.text),
            max_gens: generations.value,
            max_epochs: maxEpochs.value,
            early_stopping_patience: patience.value,
            solver_path: solverPath.text.length ? solverPath.text : null,
            book_path: bookPath.text.length ? bookPath.text : null,
            solutions_path: solutionsPath.text
        }
        Jobs.submit("training", JSON.stringify(config), "Train " + trainingDirectory.text)
    }

    function applyPreset(name) {
        if (name === "CPU") {
            device.currentIndex = device.find("cpu")
            selfPlayGames.value = 100; trainingMcts.value = 100
            selfPlayBatch.value = 64; trainBatch.value = 128
            filters.value = 16; policyLayers.value = 2; valueLayers.value = 1
            generations.value = 3
        } else if (name === "Balanced") {
            selfPlayGames.value = 500; trainingMcts.value = 600
            selfPlayBatch.value = 512; trainBatch.value = 512
            generations.value = 5
        } else {
            selfPlayGames.value = 1700; trainingMcts.value = 1400
            selfPlayBatch.value = 2000; trainBatch.value = 2000
            filters.value = 32; policyLayers.value = 4; valueLayers.value = 2
            generations.value = 10
        }
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
                        subtitle: "Generate self-play games, train the next network generation, and optionally score it with the perfect solver."
                    }

                    Text { text: "PRESET"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                    Row {
                        spacing: 8
                        Button { text: "CPU"; onClicked: page.applyPreset("CPU") }
                        Button { text: "Balanced"; onClicked: page.applyPreset("Balanced") }
                        Button { text: "GPU"; highlighted: true; onClicked: page.applyPreset("GPU") }
                    }

                    GridLayout {
                        width: parent.width
                        columns: 2
                        columnSpacing: 14
                        rowSpacing: 12
                        LabeledField { id: trainingDirectory; Layout.fillWidth: true; label: "Training directory"; text: App.trainingDir; onTextChanged: if (!activeFocus) App.setTrainingDir(text) }
                        Column {
                            Layout.fillWidth: true
                            spacing: 6
                            Text { text: "Device"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12; font.weight: Font.Medium }
                            ComboBox { id: device; width: parent.width; height: 44; model: App.cudaAvailable ? ["cuda", "cpu"] : ["cpu"]; currentIndex: 0 }
                        }
                        Column {
                            Layout.fillWidth: true; spacing: 6
                            Text { text: "Target generations"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                            SpinBox { id: generations; width: parent.width; height: 44; from: 1; to: 10000; value: 10; editable: true }
                        }
                        Column {
                            Layout.fillWidth: true; spacing: 6
                            Text { text: "Self-play games / generation"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                            SpinBox { id: selfPlayGames; width: parent.width; height: 44; from: 1; to: 10000000; value: 1700; editable: true }
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
                                SpinBox { id: selfPlayBatch; width: parent.width; from: 1; to: 1000000; value: 2000; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Training batch"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: trainBatch; width: parent.width; from: 1; to: 1000000; value: 2000; editable: true }
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
                            Column { Layout.fillWidth: true
                                Text { text: "Max epochs"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: maxEpochs; width: parent.width; from: 1; to: 10000; value: 100; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Early-stop patience"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: patience; width: parent.width; from: 0; to: 1000; value: 10; editable: true }
                            }
                        }
                        LabeledField { id: learningSchedule; width: parent.width; label: "Learning-rate schedule (generation, rate pairs)"; text: "0, 0.002, 10, 0.0008" }
                        LabeledField { id: l2; width: parent.width; label: "L2 regularization"; text: "0.0004"; validator: DoubleValidator { bottom: 0 } }

                        Text { text: "OPTIONAL PERFECT SOLVER"; color: ApplicationWindow.window.accentColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                        LabeledField { id: solverPath; width: parent.width; label: "Solver executable"; text: App.solverPath; placeholderText: "Leave empty to skip scoring" }
                        LabeledField { id: bookPath; width: parent.width; label: "Opening book"; text: App.bookPath }
                        LabeledField { id: solutionsPath; width: parent.width; label: "Solution cache"; text: App.solutionsPath }
                    }

                    Row {
                        spacing: 10
                        Button { text: Jobs.active ? "Queue training" : "Start training"; highlighted: true; onClicked: page.submitTraining() }
                        Text { anchors.verticalCenter: parent.verticalCenter; text: Jobs.active ? "This run will start after " + Jobs.currentTitle : "Artifacts are saved after each completed generation"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
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
                ProgressBar { Layout.fillWidth: true; from: 0; to: 1; value: Jobs.progress; indeterminate: Jobs.active && Jobs.progress <= 0 }
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Button {
                        Layout.fillWidth: true
                        Layout.minimumWidth: 0
                        Layout.preferredHeight: 40
                        text: Jobs.stopAfterGenerationRequested ? "Stop scheduled" : (page.compact ? "Stop" : "Stop after generation")
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
                    TextArea {
                        id: trainingLog
                        objectName: "trainingLog"
                        property real viewportWidth: Math.max(0, trainingLog.ScrollView.view ? trainingLog.ScrollView.view.width : 0)
                        width: Jobs.logs.length ? Math.max(viewportWidth, contentWidth) : viewportWidth
                        height: Math.max(implicitHeight, trainingLog.ScrollView.view ? trainingLog.ScrollView.view.height : 0)
                        text: Jobs.logs.length ? Jobs.logs : "Live phases, metrics, and worker output will appear here."
                        color: ApplicationWindow.window.mutedTextColor
                        font.family: "monospace"
                        font.pixelSize: 10
                        readOnly: true
                        selectByMouse: true
                        wrapMode: TextEdit.NoWrap
                        onTextChanged: cursorPosition = length
                        background: Rectangle { color: ApplicationWindow.window.inputColor; radius: 9 }
                    }
                }
                Text { visible: Jobs.result.length > 0; text: Jobs.result; color: ApplicationWindow.window.textColor; font.family: "monospace"; font.pixelSize: 10; wrapMode: Text.WrapAnywhere; Layout.fillWidth: true }
            }
        }
    }
}
