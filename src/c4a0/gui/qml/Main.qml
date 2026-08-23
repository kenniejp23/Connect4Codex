import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Material
import QtQuick.Layouts

ApplicationWindow {
    id: rootWindow
    width: 1440
    height: 900
    minimumWidth: 1024
    minimumHeight: 720
    visible: true
    title: "c4a0 — Connect Four AlphaZero"
    color: backgroundColor

    Material.theme: App.theme === "light" ? Material.Light : Material.Dark
    Material.accent: accentColor

    readonly property bool dark: App.theme !== "light"
    readonly property color backgroundColor: dark ? "#0A0F1D" : "#F3F5F8"
    readonly property color navColor: dark ? "#0D1424" : "#FFFFFF"
    readonly property color panelColor: dark ? "#111A2D" : "#FFFFFF"
    readonly property color panelRaisedColor: dark ? "#18243A" : "#EDF1F6"
    readonly property color inputColor: dark ? "#0D1627" : "#F7F9FB"
    readonly property color borderColor: dark ? "#26344C" : "#D8DEE8"
    readonly property color hoverColor: dark ? "#17233A" : "#EEF2F7"
    readonly property color textColor: dark ? "#F4F7FB" : "#182033"
    readonly property color mutedTextColor: dark ? "#9DAAC0" : "#5C687A"
    readonly property color faintTextColor: dark ? "#6C7890" : "#8A95A5"
    readonly property color accentColor: "#F4C44E"
    readonly property color accentMuted: dark ? "#3A321D" : "#FFF4CF"
    readonly property color successColor: "#42D3A0"
    readonly property color dangerColor: "#FF6B78"
    property bool pendingExit: false
    property string toastText: ""

    Shortcut { sequence: "Ctrl+1"; onActivated: App.setPage(0) }
    Shortcut { sequence: "Ctrl+2"; onActivated: App.setPage(1) }
    Shortcut { sequence: "Ctrl+3"; onActivated: App.setPage(2) }
    Shortcut { sequence: "Ctrl+4"; onActivated: App.setPage(3) }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.preferredWidth: rootWindow.width < 1200 ? 176 : 224
            Layout.fillHeight: true
            color: rootWindow.navColor
            border.color: rootWindow.borderColor
            border.width: 0

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 16
                spacing: 6

                Row {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 64
                    spacing: 12
                    Rectangle {
                        width: 42; height: 42; radius: 12
                        color: rootWindow.accentColor
                        Text {
                            anchors.centerIn: parent
                            text: "C4"
                            color: "#111827"
                            font.pixelSize: 15
                            font.weight: Font.Bold
                        }
                    }
                    Column {
                        anchors.verticalCenter: parent.verticalCenter
                        Text { text: "c4a0"; color: rootWindow.textColor; font.pixelSize: 19; font.weight: Font.Bold }
                        Text { text: "ALPHAZERO LAB"; color: rootWindow.faintTextColor; font.pixelSize: 9; font.letterSpacing: 1.2 }
                    }
                }

                NavButton { Layout.fillWidth: true; text: "Home"; shortLabel: "01"; selected: App.page === 0; onClicked: App.setPage(0) }
                NavButton { Layout.fillWidth: true; text: "Play"; shortLabel: "02"; selected: App.page === 1; onClicked: App.setPage(1) }
                NavButton { Layout.fillWidth: true; text: "Training"; shortLabel: "03"; selected: App.page === 2; onClicked: App.setPage(2) }
                NavButton { Layout.fillWidth: true; text: "Evaluation"; shortLabel: "04"; selected: App.page === 3; onClicked: App.setPage(3) }
                NavButton { Layout.fillWidth: true; text: "Models & Data"; shortLabel: "05"; selected: App.page === 4; onClicked: App.setPage(4) }
                NavButton { Layout.fillWidth: true; text: "Validation"; shortLabel: "06"; selected: App.page === 5; onClicked: App.setPage(5) }
                NavButton { Layout.fillWidth: true; text: "Settings"; shortLabel: "07"; selected: App.page === 6; onClicked: App.setPage(6) }

                Item { Layout.fillHeight: true }

                Panel {
                    Layout.fillWidth: true
                    Layout.preferredHeight: Jobs.active ? 118 : 82
                    color: Jobs.active ? rootWindow.panelRaisedColor : rootWindow.panelColor
                    Column {
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 7
                        Row {
                            width: parent.width
                            spacing: 8
                            Rectangle {
                                width: 8; height: 8; radius: 4
                                anchors.verticalCenter: parent.verticalCenter
                                color: Jobs.active ? rootWindow.successColor : rootWindow.faintTextColor
                                SequentialAnimation on opacity {
                                    running: Jobs.active && !App.reducedMotion
                                    loops: Animation.Infinite
                                    NumberAnimation { to: 0.3; duration: 650 }
                                    NumberAnimation { to: 1.0; duration: 650 }
                                }
                            }
                            Text {
                                width: parent.width - 20
                                text: Jobs.active ? Jobs.currentTitle : "Compute queue idle"
                                elide: Text.ElideRight
                                color: rootWindow.textColor
                                font.pixelSize: 12
                                font.weight: Font.DemiBold
                            }
                        }
                        Text { text: Jobs.active ? Jobs.phase : "Ready for work"; color: rootWindow.mutedTextColor; font.pixelSize: 11 }
                        ProgressBar { visible: Jobs.active; width: parent.width; from: 0; to: 1; value: Jobs.progress; indeterminate: Jobs.progress <= 0 }
                        Text { visible: Jobs.queueDepth > 0; text: Jobs.queueDepth + " queued"; color: rootWindow.accentColor; font.pixelSize: 10 }
                    }
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: 0

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: rootWindow.width < 1120 ? 64 : 72
                color: rootWindow.backgroundColor
                RowLayout {
                    anchors.fill: parent
                    anchors.leftMargin: rootWindow.width < 1120 ? 20 : 28
                    anchors.rightMargin: rootWindow.width < 1120 ? 20 : 28
                    Text {
                        text: ["Overview", "Play", "Training", "Evaluation", "Models & Data", "Validation", "Settings"][App.page]
                        color: rootWindow.textColor
                        font.pixelSize: 23
                        font.weight: Font.DemiBold
                    }
                    Item { Layout.fillWidth: true }
                    Rectangle {
                        Layout.preferredWidth: deviceText.implicitWidth + 26
                        Layout.preferredHeight: 34
                        radius: 17
                        color: rootWindow.panelColor
                        border.color: rootWindow.borderColor
                        Text {
                            id: deviceText
                            anchors.centerIn: parent
                            text: "DEVICE  " + App.device.toUpperCase()
                            color: rootWindow.mutedTextColor
                            font.pixelSize: 10
                            font.weight: Font.DemiBold
                            font.letterSpacing: 0.8
                        }
                    }
                }
            }

            StackLayout {
                Layout.fillWidth: true
                Layout.fillHeight: true
                currentIndex: App.page
                HomePage {}
                PlayPage {}
                TrainingPage {}
                EvaluationPage {}
                ModelsPage {}
                ValidationPage {}
                SettingsPage {}
            }
        }
    }

    Dialog {
        id: closeDialog
        anchors.centerIn: parent
        modal: true
        title: "A job is still running"
        standardButtons: Dialog.Cancel | Dialog.Ok
        onAccepted: {
            rootWindow.pendingExit = true
            Jobs.cancel()
        }
        Text {
            text: "Cancel the active job and close after it has stopped?"
            color: rootWindow.textColor
        }
    }

    Connections {
        target: Jobs
        function onNotification(message) {
            rootWindow.toastText = message
            toast.visible = true
            toastTimer.restart()
        }
    }

    Rectangle {
        id: toast
        z: 100
        visible: false
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 24
        width: Math.min(520, Math.max(280, toastLabel.implicitWidth + 40))
        height: toastLabel.implicitHeight + 28
        radius: 12
        color: rootWindow.panelRaisedColor
        border.color: rootWindow.borderColor
        Text {
            id: toastLabel
            anchors.fill: parent
            anchors.margins: 14
            text: rootWindow.toastText
            color: rootWindow.textColor
            font.pixelSize: 12
            wrapMode: Text.WordWrap
        }
    }

    Timer {
        id: toastTimer
        interval: 5000
        onTriggered: toast.visible = false
    }

    Timer {
        interval: 150
        repeat: true
        running: rootWindow.pendingExit
        onTriggered: if (!Jobs.active) Qt.quit()
    }

    onClosing: function(close) {
        if (Jobs.active && !pendingExit) {
            close.accepted = false
            closeDialog.open()
        }
    }
}
