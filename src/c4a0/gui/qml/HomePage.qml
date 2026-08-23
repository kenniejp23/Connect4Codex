import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ScrollView {
        id: homeScroll
        anchors.fill: parent
        anchors.margins: 28
        clip: true
        Column {
            width: homeScroll.availableWidth
            spacing: 20

            SectionTitle {
                width: parent.width
                title: "Welcome back"
                subtitle: "Play, train, and evaluate Connect Four agents from one focused workspace."
            }

            Row {
                width: parent.width
                spacing: 16
                Repeater {
                    model: [
                        { label: "COMPUTE", value: App.device.toUpperCase(), note: App.cudaAvailable ? "CUDA acceleration ready" : "CPU execution" },
                        { label: "LATEST MODEL", value: App.latestGeneration, note: App.trainingDir },
                        { label: "JOB QUEUE", value: Jobs.active ? "Running" : "Idle", note: Jobs.queueDepth + " waiting" }
                    ]
                    Panel {
                        width: (parent.width - 32) / 3
                        height: 132
                        Column {
                            anchors.fill: parent
                            anchors.margins: 18
                            spacing: 9
                            Text { text: modelData.label; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1.1 }
                            Text { text: modelData.value; color: ApplicationWindow.window.textColor; font.pixelSize: 22; font.weight: Font.DemiBold; elide: Text.ElideRight; width: parent.width }
                            Text { text: modelData.note; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12; elide: Text.ElideMiddle; width: parent.width }
                        }
                    }
                }
            }

            Panel {
                width: parent.width
                height: 210
                RowLayout {
                    anchors.fill: parent
                    anchors.margins: 22
                    spacing: 24
                    ColumnLayout {
                        Layout.fillWidth: true
                        spacing: 9
                        Text { text: "Start with a game"; color: ApplicationWindow.window.textColor; font.pixelSize: 22; font.weight: Font.DemiBold }
                        Text { Layout.fillWidth: true; text: "Challenge the latest model, play locally, or watch two agents reason in real time."; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 13; wrapMode: Text.WordWrap }
                        Button { text: "Open Play"; onClicked: App.setPage(1) }
                    }
                    Rectangle {
                        Layout.preferredWidth: 230
                        Layout.fillHeight: true
                        radius: 12
                        color: "#164A83"
                        Grid {
                            anchors.centerIn: parent
                            rows: 3; columns: 4; spacing: 8
                            Repeater {
                                model: 12
                                Rectangle {
                                    width: 31; height: 31; radius: 16
                                    color: index % 3 === 0 ? "#EF5B67" : (index % 4 === 0 ? "#F4C44E" : "#0C2D52")
                                    border.color: "#FFFFFF22"
                                }
                            }
                        }
                    }
                }
            }

            Text { text: "RECENT ACTIVITY"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1.1 }
            Panel {
                width: parent.width
                height: Math.max(90, recentColumn.height + 32)
                Column {
                    id: recentColumn
                    anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top
                    anchors.margins: 16
                    spacing: 8
                    Text {
                        visible: Jobs.recentJobs.length === 0
                        text: "Completed training, evaluation, and validation jobs will appear here."
                        color: ApplicationWindow.window.mutedTextColor
                        font.pixelSize: 13
                    }
                    Repeater {
                        model: Jobs.recentJobs
                        Rectangle {
                            width: recentColumn.width
                            height: 46
                            radius: 8
                            color: ApplicationWindow.window.panelRaisedColor
                            RowLayout {
                                anchors.fill: parent; anchors.leftMargin: 12; anchors.rightMargin: 12
                                Text { Layout.fillWidth: true; text: modelData.title; color: ApplicationWindow.window.textColor; font.pixelSize: 12; font.weight: Font.Medium }
                                Text { text: modelData.status; color: modelData.status === "Completed" ? ApplicationWindow.window.successColor : ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                            }
                        }
                    }
                }
            }
        }
    }
}
