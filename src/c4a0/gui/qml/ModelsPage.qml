import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: page
    property int selectedGeneration: -1
    property string stats: "Select a generation to inspect its self-play data."
    property var details: ({})

    function selectGeneration(generation) {
        page.selectedGeneration = generation
        page.stats = App.generationStats(generation)
        try {
            page.details = JSON.parse(page.stats)
        } catch (error) {
            page.details = ({ error: String(error) })
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 28
        anchors.rightMargin: 28
        anchors.bottomMargin: 24
        spacing: 14

        Panel {
            Layout.fillWidth: true
            Layout.preferredHeight: 82
            RowLayout {
                anchors.fill: parent; anchors.margins: 14
                LabeledField { id: modelDirectory; Layout.fillWidth: true; label: "Training directory"; text: App.trainingDir }
                Button { text: "Load"; onClicked: { App.setTrainingDir(modelDirectory.text); App.refreshGenerations() } }
                Button { text: "Open in Play"; enabled: App.generations.length > 0; onClicked: App.setPage(1) }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 14
            Panel {
                Layout.fillWidth: true
                Layout.fillHeight: true
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 16; spacing: 10
                    RowLayout {
                        Layout.fillWidth: true
                        Text { text: "GENERATIONS"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                        Item { Layout.fillWidth: true }
                        Text { text: App.generations.length + " found"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                    }
                    Rectangle { Layout.fillWidth: true; Layout.preferredHeight: 36; radius: 7; color: ApplicationWindow.window.panelRaisedColor
                        RowLayout { anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12
                            Text { Layout.preferredWidth: 90; text: "Generation"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10 }
                            Text { Layout.fillWidth: true; text: "Created"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10 }
                            Text { Layout.preferredWidth: 100; text: "Validation loss"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; horizontalAlignment: Text.AlignRight }
                            Text { Layout.preferredWidth: 90; text: "Solver"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; horizontalAlignment: Text.AlignRight }
                        }
                    }
                    ListView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        clip: true
                        spacing: 6
                        model: App.generations
                        delegate: Rectangle {
                            required property var modelData
                            width: ListView.view.width
                            height: 48
                            radius: 8
                            color: page.selectedGeneration === modelData.generation
                                   ? (ApplicationWindow.window ? ApplicationWindow.window.accentMuted : "#3A321D")
                                   : (ApplicationWindow.window ? ApplicationWindow.window.inputColor : "#0D1627")
                            border.color: page.selectedGeneration === modelData.generation
                                          ? (ApplicationWindow.window ? ApplicationWindow.window.accentColor : "#F4C44E")
                                          : "transparent"
                            RowLayout { anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12
                                Text { Layout.preferredWidth: 90; text: "Gen " + modelData.generation; color: ApplicationWindow.window ? ApplicationWindow.window.textColor : "#F4F7FB"; font.pixelSize: 12; font.weight: Font.DemiBold }
                                Text { Layout.fillWidth: true; text: modelData.created; color: ApplicationWindow.window ? ApplicationWindow.window.mutedTextColor : "#9DAAC0"; font.pixelSize: 11 }
                                Text { Layout.preferredWidth: 100; text: modelData.valLoss === null || modelData.valLoss === undefined ? "—" : Number(modelData.valLoss).toFixed(4); color: ApplicationWindow.window ? ApplicationWindow.window.textColor : "#F4F7FB"; font.pixelSize: 11; horizontalAlignment: Text.AlignRight }
                                Text { Layout.preferredWidth: 90; text: modelData.solverScore === null || modelData.solverScore === undefined ? "—" : (Number(modelData.solverScore) * 100).toFixed(1) + "%"; color: ApplicationWindow.window ? ApplicationWindow.window.textColor : "#F4F7FB"; font.pixelSize: 11; horizontalAlignment: Text.AlignRight }
                            }
                            MouseArea { anchors.fill: parent; cursorShape: Qt.PointingHandCursor; onClicked: page.selectGeneration(modelData.generation) }
                        }
                        Label { anchors.centerIn: parent; visible: App.generations.length === 0; text: "No generations found in this directory"; color: ApplicationWindow.window.mutedTextColor }
                    }
                }
            }

            Panel {
                Layout.preferredWidth: 360
                Layout.fillHeight: true
                ColumnLayout {
                    anchors.fill: parent; anchors.margins: 16; spacing: 12
                    Text { text: page.selectedGeneration < 0 ? "GENERATION INSPECTOR" : "GENERATION " + page.selectedGeneration; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                    Text { text: "Self-play statistics"; color: ApplicationWindow.window.textColor; font.pixelSize: 18; font.weight: Font.DemiBold }
                    RowLayout {
                        Layout.fillWidth: true
                        visible: page.details.games !== undefined
                        Repeater {
                            model: [
                                { label: "GAMES", value: page.details.games || 0 },
                                { label: "SAMPLES", value: page.details.samples || 0 },
                                { label: "UNIQUE", value: page.details.uniquePositions || 0 }
                            ]
                            ColumnLayout {
                                Layout.fillWidth: true
                                Text { text: modelData.label; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9 }
                                Text { text: modelData.value; color: ApplicationWindow.window.textColor; font.pixelSize: 17; font.weight: Font.DemiBold }
                            }
                        }
                    }
                    Text { visible: page.details.games !== undefined; text: "OUTCOMES"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 9; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                    Repeater {
                        model: page.details.games === undefined ? [] : [
                            { label: "Red", value: page.details.redWins || 0, color: ApplicationWindow.window.dangerColor },
                            { label: "Draw", value: page.details.draws || 0, color: ApplicationWindow.window.mutedTextColor },
                            { label: "Gold", value: page.details.goldWins || 0, color: ApplicationWindow.window.accentColor }
                        ]
                        RowLayout {
                            Layout.fillWidth: true
                            Text { Layout.preferredWidth: 36; text: modelData.label; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 10 }
                            Rectangle {
                                Layout.fillWidth: true; Layout.preferredHeight: 8; radius: 4; color: ApplicationWindow.window.inputColor
                                Rectangle { width: parent.width * modelData.value / Math.max(1, page.details.games); height: parent.height; radius: 4; color: modelData.color }
                            }
                            Text { Layout.preferredWidth: 28; text: modelData.value; color: ApplicationWindow.window.textColor; font.pixelSize: 10; horizontalAlignment: Text.AlignRight }
                        }
                    }
                    Text { visible: page.details.averageMoves !== undefined; text: "Average game length  " + Number(page.details.averageMoves || 0).toFixed(1) + " moves"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                    TextArea {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        text: page.stats
                        color: ApplicationWindow.window.mutedTextColor
                        font.family: "monospace"
                        font.pixelSize: 11
                        readOnly: true
                        wrapMode: TextEdit.WrapAnywhere
                        background: Rectangle { color: ApplicationWindow.window.inputColor; radius: 9 }
                    }
                    Button { Layout.fillWidth: true; text: "Use this model in Play"; enabled: page.selectedGeneration >= 0; highlighted: true; onClicked: App.useGenerationInPlay(page.selectedGeneration) }
                }
            }
        }
    }
}
