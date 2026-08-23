import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ScrollView {
        anchors.fill: parent
        anchors.leftMargin: 28
        anchors.rightMargin: 28
        anchors.bottomMargin: 24
        clip: true
        Column {
            width: parent.width
            spacing: 16
            Panel {
                width: parent.width
                height: appearance.height + 40
                Column {
                    id: appearance
                    anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 20
                    spacing: 14
                    SectionTitle { width: parent.width; title: "Appearance"; subtitle: "The technical dark theme is optimized for long training and analysis sessions." }
                    RowLayout {
                        width: parent.width
                        Text { Layout.fillWidth: true; text: "Color theme"; color: ApplicationWindow.window.textColor; font.pixelSize: 13 }
                        ComboBox { model: ["Dark", "Light"]; currentIndex: App.theme === "light" ? 1 : 0; onActivated: App.setTheme(currentIndex === 1 ? "light" : "dark") }
                    }
                    RowLayout {
                        width: parent.width
                        ColumnLayout { Layout.fillWidth: true
                            Text { text: "Reduce motion"; color: ApplicationWindow.window.textColor; font.pixelSize: 13 }
                            Text { text: "Replace counter drops and winning pulses with static state changes."; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                        }
                        Switch { checked: App.reducedMotion; onToggled: App.setReducedMotion(checked) }
                    }
                }
            }
            Panel {
                width: parent.width
                height: paths.height + 40
                Column {
                    id: paths
                    anchors.left: parent.left; anchors.right: parent.right; anchors.top: parent.top; anchors.margins: 20
                    spacing: 14
                    SectionTitle { width: parent.width; title: "Default paths"; subtitle: "These values prefill Play, Training, and Evaluation forms." }
                    LabeledField { id: settingsTraining; width: parent.width; label: "Training directory"; text: App.trainingDir }
                    LabeledField { id: settingsSolver; width: parent.width; label: "Solver executable"; text: App.solverPath }
                    LabeledField { id: settingsBook; width: parent.width; label: "Opening book"; text: App.bookPath }
                    LabeledField { id: settingsCache; width: parent.width; label: "Solutions cache"; text: App.solutionsPath }
                    Button {
                        text: "Save paths"
                        highlighted: true
                        onClicked: {
                            App.setTrainingDir(settingsTraining.text)
                            App.setSolverPaths(settingsSolver.text, settingsBook.text, settingsCache.text)
                        }
                    }
                }
            }
            Panel {
                width: parent.width
                height: 110
                RowLayout {
                    anchors.fill: parent; anchors.margins: 20
                    ColumnLayout {
                        Layout.fillWidth: true
                        Text { text: "Runtime"; color: ApplicationWindow.window.textColor; font.pixelSize: 18; font.weight: Font.DemiBold }
                        Text { text: "Detected device: " + App.device + "  •  Linux-first desktop build  •  Native engine ready"; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12 }
                    }
                    Rectangle { width: 10; height: 10; radius: 5; color: ApplicationWindow.window.successColor }
                }
            }
        }
    }
}
