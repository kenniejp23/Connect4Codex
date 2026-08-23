import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    ColumnLayout {
        anchors.fill: parent
        anchors.leftMargin: 28
        anchors.rightMargin: 28
        anchors.bottomMargin: 24
        spacing: 14
        Panel {
            Layout.fillWidth: true
            Layout.preferredHeight: 126
            RowLayout {
                anchors.fill: parent; anchors.margins: 18; spacing: 16
                ColumnLayout {
                    Layout.fillWidth: true
                    Text { text: "Developer validation"; color: ApplicationWindow.window.textColor; font.pixelSize: 20; font.weight: Font.DemiBold }
                    Text { text: App.sourceCheckout ? "Run the same lint, type, native, Python, and CI profiles used by the project." : "Validation is available only from a source checkout containing mise.toml."; color: ApplicationWindow.window.mutedTextColor; font.pixelSize: 12; wrapMode: Text.WordWrap; Layout.fillWidth: true }
                }
                ComboBox { id: profile; Layout.preferredWidth: 180; model: ["lint", "typecheck", "test:cpp", "test:python", "check", "ci"]; currentIndex: 4 }
                Button {
                    text: Jobs.active ? "Queue validation" : "Run validation"
                    highlighted: true
                    enabled: App.sourceCheckout
                    onClicked: Jobs.submit("validation", JSON.stringify({ profile: profile.currentText, project_dir: "." }), "Validation: " + profile.currentText)
                }
            }
        }
        Panel {
            Layout.fillWidth: true
            Layout.fillHeight: true
            ColumnLayout {
                anchors.fill: parent; anchors.margins: 16; spacing: 10
                RowLayout {
                    Layout.fillWidth: true
                    Text { text: "OUTPUT"; color: ApplicationWindow.window.faintTextColor; font.pixelSize: 10; font.weight: Font.DemiBold; font.letterSpacing: 1 }
                    Item { Layout.fillWidth: true }
                    Text { text: Jobs.phase; color: Jobs.active ? ApplicationWindow.window.successColor : ApplicationWindow.window.mutedTextColor; font.pixelSize: 11 }
                    Button { text: "Cancel"; visible: Jobs.active; onClicked: Jobs.cancel() }
                    Button { text: "Clear"; flat: true; onClicked: Jobs.clearLogs() }
                }
                TextArea {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    text: Jobs.logs.length ? Jobs.logs : "Choose a profile and run it to stream output here."
                    color: ApplicationWindow.window.mutedTextColor
                    font.family: "monospace"
                    font.pixelSize: 10
                    readOnly: true
                    wrapMode: TextEdit.WrapAnywhere
                    background: Rectangle { color: ApplicationWindow.window.inputColor; radius: 9 }
                }
                Text { visible: Jobs.result.length > 0; text: Jobs.result; color: ApplicationWindow.window.textColor; font.family: "monospace"; font.pixelSize: 10; wrapMode: Text.WrapAnywhere; Layout.fillWidth: true }
            }
        }
    }
}
