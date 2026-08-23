import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    objectName: "evaluationPage"

    function numberList(text) {
        return text.split(",").map(function(value) { return Number(value.trim()) })
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 28
        anchors.rightMargin: 28
        anchors.bottomMargin: 24
        spacing: 14

        TabBar {
            id: tabs
            Layout.fillWidth: true
            TabButton { text: "Tournament" }
            TabButton { text: "Solver scoring" }
            TabButton { text: "Neural sweep" }
            TabButton { text: "MCTS sweep" }
        }

        StackLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: tabs.currentIndex

            ScrollView {
                id: tournamentScroll
                objectName: "tournamentScrollView"
                clip: true
                contentWidth: tournamentPanel.width
                contentHeight: tournamentPanel.height
                ScrollBar.vertical.policy: ScrollBar.AsNeeded
                ScrollBar.horizontal.policy: ScrollBar.AsNeeded
                Panel {
                    id: tournamentPanel
                    objectName: "tournamentPanel"
                    width: Math.max(page.width - 56, 1110)
                    height: tournamentForm.implicitHeight + 40
                    Column {
                        id: tournamentForm
                        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 20
                        spacing: 14
                        RowLayout {
                            width: parent.width
                            spacing: 18
                            SectionTitle {
                                Layout.fillWidth: true
                                title: "Round-robin tournament"
                                subtitle: "Compare trained generations and baseline players with both seat orders."
                            }
                            Button {
                                Layout.alignment: Qt.AlignTop
                                Layout.preferredWidth: 180
                                leftPadding: 12
                                rightPadding: 12
                                text: Jobs.active ? "Queue tournament" : "Run tournament"
                                highlighted: true
                                onClicked: Jobs.submit("tournament", JSON.stringify({
                                    players: tournamentPlayers.text.split(",").map(function(v) { return v.trim() }),
                                    base_dir: tournamentBase.text, device: App.device,
                                    games_per_match: tournamentGames.value, batch_size: tournamentBatch.value,
                                    mcts_iterations: tournamentMcts.value,
                                    exploration_constant: Number(tournamentExploration.text),
                                    c_ply_penalty: Number(tournamentPly.text)
                                }), "Tournament")
                            }
                        }
                        LabeledField { id: tournamentPlayers; width: parent.width; label: "Players (latest, random, uniform, or gen:N)"; text: "latest, random, uniform" }
                        RowLayout {
                            width: parent.width
                            spacing: 10
                            LabeledField { id: tournamentBase; Layout.preferredWidth: 260; Layout.minimumWidth: 260; label: "Training directory"; text: App.trainingDir }
                            Column { Layout.preferredWidth: 145; Layout.minimumWidth: 145
                                Text { text: "Games / match (even)"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: tournamentGames; objectName: "tournamentGames"; width: parent.width; from: 2; to: 10000; stepSize: 2; value: 2; editable: true }
                            }
                            Column { Layout.preferredWidth: 130; Layout.minimumWidth: 130
                                Text { text: "Batch size"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: tournamentBatch; width: parent.width; from: 1; to: 100000; value: 64; editable: true }
                            }
                            Column { Layout.preferredWidth: 260; Layout.minimumWidth: 260
                                Text { text: "MCTS iterations"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: tournamentMcts; width: parent.width; from: 1; to: 100000; value: 200; editable: true }
                            }
                            LabeledField { id: tournamentExploration; Layout.preferredWidth: 135; Layout.minimumWidth: 135; label: "Exploration constant"; text: "6.6"; validator: DoubleValidator { bottom: 0 } }
                            LabeledField { id: tournamentPly; Layout.preferredWidth: 130; Layout.minimumWidth: 130; label: "Ply penalty"; text: "0.01"; validator: DoubleValidator { bottom: 0 } }
                        }
                        Column {
                            width: parent.width
                            spacing: 8
                            visible: Jobs.resultData.ranking !== undefined
                            Text { text: "RANKING"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                            Rectangle {
                                width: parent.width
                                height: 26
                                radius: 7
                                color: ApplicationWindow.window.panelRaisedColor
                                RowLayout {
                                    anchors.fill: parent
                                    anchors.leftMargin: 12
                                    anchors.rightMargin: 12
                                    Text { Layout.preferredWidth: 34; text: "#"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold }
                                    Text { Layout.fillWidth: true; text: "PLAYER"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold }
                                    Text { Layout.preferredWidth: 116; text: "RECORD"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; horizontalAlignment: Text.AlignRight }
                                    Text { Layout.preferredWidth: 68; text: "POINTS"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; horizontalAlignment: Text.AlignRight }
                                }
                            }
                            Repeater {
                                model: Jobs.resultData.ranking || []
                                Rectangle {
                                    width: parent.width
                                    height: 38
                                    radius: 8
                                    color: index % 2 === 0 ? ApplicationWindow.window.inputColor : ApplicationWindow.window.panelColor
                                    RowLayout {
                                        anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12
                                        Text { Layout.preferredWidth: 34; text: index + 1; color: ApplicationWindow.window.accentColor; font.pixelSize: 11; font.weight: Font.Bold }
                                        Text { Layout.fillWidth: true; text: modelData.player; color: ApplicationWindow.window.textColor; font.pixelSize: 12; font.weight: Font.DemiBold }
                                        Text { Layout.preferredWidth: 116; text: modelData.wins + " W   " + modelData.draws + " D   " + modelData.losses + " L"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11; horizontalAlignment: Text.AlignRight }
                                        Text { Layout.preferredWidth: 68; text: Number(modelData.points).toFixed(1) + " pts"; color: ApplicationWindow.window.textColor; font.pixelSize: 11; horizontalAlignment: Text.AlignRight }
                                    }
                                }
                            }
                            Text { text: "MATCHUPS (W-D-L)"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                            Repeater {
                                model: Jobs.resultData.matchups || []
                                RowLayout {
                                    width: parent.width
                                    spacing: 10
                                    Text { Layout.preferredWidth: 116; text: modelData.player; color: ApplicationWindow.window.textColor; font.pixelSize: 11; font.weight: Font.DemiBold }
                                    Repeater {
                                        model: modelData.opponents
                                        Rectangle {
                                            Layout.fillWidth: true
                                    Layout.preferredHeight: 24
                                            radius: 6
                                            color: ApplicationWindow.window.inputColor
                                            Text {
                                                anchors.fill: parent
                                                anchors.leftMargin: 10
                                                anchors.rightMargin: 10
                                                verticalAlignment: Text.AlignVCenter
                                                text: modelData.opponent + "   " + modelData.wins + "–" + modelData.draws + "–" + modelData.losses
                                                color: ApplicationWindow.window.mutedTextColor
                                                font.pixelSize: 11
                                                elide: Text.ElideRight
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            ScrollView {
                clip: true
                Panel {
                    width: parent.width
                    height: solverForm.height + 40
                    Column {
                        id: solverForm
                        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 20
                        spacing: 16
                        SectionTitle { width: parent.width; title: "Perfect-solver scoring"; subtitle: "Measure saved policies against objective Connect Four solutions." }
                        GridLayout {
                            width: parent.width; columns: 2; columnSpacing: 14; rowSpacing: 12
                            LabeledField { id: scoreSolver; Layout.fillWidth: true; label: "Solver executable"; text: App.solverPath }
                            LabeledField { id: scoreBook; Layout.fillWidth: true; label: "Opening book"; text: App.bookPath }
                            LabeledField { id: scoreBase; Layout.fillWidth: true; label: "Training directory"; text: App.trainingDir }
                            LabeledField { id: scoreCache; Layout.fillWidth: true; label: "Solutions cache"; text: App.solutionsPath }
                            LabeledField { id: scoreGenerations; Layout.fillWidth: true; label: "Generations (blank selects all)"; placeholderText: "0, 4, 8" }
                        }
                        CheckBox { id: rescore; text: "Rescore generations that already have a solver score" }
                        Button {
                            text: Jobs.active ? "Queue scoring" : "Start scoring"
                            highlighted: true
                            enabled: scoreSolver.text.length > 0 && scoreBook.text.length > 0
                            onClicked: Jobs.submit("solver_score", JSON.stringify({
                                solver_path: scoreSolver.text, book_path: scoreBook.text,
                                base_dir: scoreBase.text, solutions_path: scoreCache.text,
                                rescore: rescore.checked,
                                generations: scoreGenerations.text.trim().length ? page.numberList(scoreGenerations.text) : null
                            }), "Solver scoring")
                        }
                    }
                }
            }

            ScrollView {
                clip: true
                Panel {
                    width: parent.width
                    height: nnForm.height + 40
                    Column {
                        id: nnForm
                        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 20
                        spacing: 16
                        SectionTitle { width: parent.width; title: "Neural-network hyperparameter sweep"; subtitle: "Search architecture and optimizer ranges against existing self-play data." }
                        GridLayout {
                            width: parent.width; columns: 3; columnSpacing: 14; rowSpacing: 12
                            LabeledField { id: nnBase; Layout.fillWidth: true; label: "Training directory"; text: App.trainingDir }
                            LabeledField { id: nnStudy; Layout.fillWidth: true; label: "Study name / database stem"; text: "sweep_hparam" }
                            Column { Layout.fillWidth: true
                                Text { text: "Generations sampled"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: nnGens; width: parent.width; from: 1; to: 1000; value: 5; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Trials"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: nnTrials; width: parent.width; from: 1; to: 10000; value: 100; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Max epochs / trial"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: nnEpochs; width: parent.width; from: 1; to: 10000; value: 30; editable: true }
                            }
                            LabeledField { id: nnBatches; Layout.fillWidth: true; label: "Batch sizes"; text: "256, 512, 1024" }
                            LabeledField { id: nnResidual; Layout.fillWidth: true; label: "Residual blocks min,max"; text: "0, 1" }
                            LabeledField { id: nnFilters; Layout.fillWidth: true; label: "Conv filters min,max"; text: "16, 64" }
                            LabeledField { id: nnPolicy; Layout.fillWidth: true; label: "Policy layers min,max"; text: "1, 4" }
                            LabeledField { id: nnValue; Layout.fillWidth: true; label: "Value layers min,max"; text: "1, 2" }
                            LabeledField { id: nnLr; Layout.fillWidth: true; label: "Learning rate min,max"; text: "0.0001, 0.01" }
                            LabeledField { id: nnL2; Layout.fillWidth: true; label: "L2 regularization min,max"; text: "0.00001, 0.001" }
                        }
                        Button {
                            text: Jobs.active ? "Queue NN sweep" : "Start NN sweep"
                            highlighted: true
                            onClicked: {
                                var rb = page.numberList(nnResidual.text), fs = page.numberList(nnFilters.text)
                                var pp = page.numberList(nnPolicy.text), vv = page.numberList(nnValue.text)
                                var lr = page.numberList(nnLr.text), l2v = page.numberList(nnL2.text)
                                Jobs.submit("nn_sweep", JSON.stringify({
                                    base_dir: nnBase.text, study_name: nnStudy.text,
                                    n_gens: nnGens.value, n_trials: nnTrials.value, max_epochs: nnEpochs.value,
                                    residual_blocks_min: rb[0], residual_blocks_max: rb[1],
                                    filter_size_min: fs[0], filter_size_max: fs[1],
                                    policy_layers_min: pp[0], policy_layers_max: pp[1],
                                    value_layers_min: vv[0], value_layers_max: vv[1],
                                    learning_rate_min: lr[0], learning_rate_max: lr[1],
                                    l2_reg_min: l2v[0], l2_reg_max: l2v[1],
                                    batch_sizes: page.numberList(nnBatches.text)
                                }), "Neural sweep")
                            }
                        }
                    }
                }
            }

            ScrollView {
                clip: true
                Panel {
                    width: parent.width
                    height: mctsForm.height + 40
                    Column {
                        id: mctsForm
                        anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 20
                        spacing: 16
                        SectionTitle { width: parent.width; title: "MCTS hyperparameter sweep"; subtitle: "Run independent training trials and maximize perfect-solver score." }
                        GridLayout {
                            width: parent.width; columns: 3; columnSpacing: 14; rowSpacing: 12
                            LabeledField { id: msBase; Layout.fillWidth: true; label: "Trial training directory"; text: "training-sweeps" }
                            LabeledField { id: msDb; Layout.fillWidth: true; label: "Optuna database"; text: "optuna.db" }
                            Column { Layout.fillWidth: true
                                Text { text: "Trials"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msTrials; width: parent.width; from: 1; to: 10000; value: 100; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Generations / trial"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msGens; width: parent.width; from: 1; to: 1000; value: 10; editable: true }
                            }
                            LabeledField { id: msGamesRange; Layout.fillWidth: true; label: "Self-play games min,max"; text: "1000, 5000" }
                            LabeledField { id: msMctsRange; Layout.fillWidth: true; label: "MCTS iterations min,max"; text: "100, 1500" }
                            LabeledField { id: msExplorationRange; Layout.fillWidth: true; label: "Exploration min,max"; text: "0.5, 12.0" }
                            Column { Layout.fillWidth: true
                                Text { text: "Inference batch"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msSelfBatch; width: parent.width; from: 1; to: 1000000; value: 2000; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Training batch"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msTrainBatch; width: parent.width; from: 1; to: 1000000; value: 2000; editable: true }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Residual blocks"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msResidual; width: parent.width; from: 0; to: 64; value: 1 }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Conv filters"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msFilters; width: parent.width; from: 1; to: 4096; value: 32 }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Policy layers"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msPolicy; width: parent.width; from: 1; to: 64; value: 4 }
                            }
                            Column { Layout.fillWidth: true
                                Text { text: "Value layers"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                                SpinBox { id: msValue; width: parent.width; from: 1; to: 64; value: 2 }
                            }
                            LabeledField { id: msSchedule; Layout.fillWidth: true; label: "Learning-rate schedule"; text: "0, 0.002" }
                            LabeledField { id: msL2; Layout.fillWidth: true; label: "L2 regularization"; text: "0.0004" }
                            LabeledField { id: msPly; Layout.fillWidth: true; label: "Ply penalty"; text: "0.01" }
                            LabeledField { id: msSolver; Layout.fillWidth: true; label: "Solver executable"; text: App.solverPath }
                            LabeledField { id: msBook; Layout.fillWidth: true; label: "Opening book"; text: App.bookPath }
                            LabeledField { id: msCache; Layout.fillWidth: true; label: "Solutions cache"; text: App.solutionsPath }
                        }
                        Button {
                            text: Jobs.active ? "Queue MCTS sweep" : "Start MCTS sweep"
                            highlighted: true
                            enabled: msSolver.text.length > 0 && msBook.text.length > 0
                            onClicked: {
                                var games = page.numberList(msGamesRange.text)
                                var mcts = page.numberList(msMctsRange.text)
                                var explore = page.numberList(msExplorationRange.text)
                                Jobs.submit("mcts_sweep", JSON.stringify({
                                    device: App.device, c_ply_penalty: Number(msPly.text),
                                    self_play_batch_size: msSelfBatch.value, training_batch_size: msTrainBatch.value,
                                    n_residual_blocks: msResidual.value, conv_filter_size: msFilters.value,
                                    n_policy_layers: msPolicy.value, n_value_layers: msValue.value,
                                    lr_schedule: page.numberList(msSchedule.text), l2_reg: Number(msL2.text),
                                    base_training_dir: msBase.text, optuna_db_path: msDb.text,
                                    n_trials: msTrials.value, max_gens_per_trial: msGens.value,
                                    solver_path: msSolver.text, book_path: msBook.text, solutions_path: msCache.text,
                                    self_play_games_min: games[0], self_play_games_max: games[1],
                                    mcts_iterations_min: mcts[0], mcts_iterations_max: mcts[1],
                                    exploration_min: explore[0], exploration_max: explore[1]
                                }), "MCTS sweep")
                            }
                        }
                    }
                }
            }
        }
    }
}
