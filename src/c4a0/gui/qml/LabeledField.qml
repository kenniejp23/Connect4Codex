import QtQuick
import QtQuick.Controls

Column {
    id: root
    property alias label: labelItem.text
    property alias text: field.text
    property alias placeholderText: field.placeholderText
    property alias validator: field.validator
    property alias inputMethodHints: field.inputMethodHints
    property alias field: field
    signal editingFinished()
    spacing: 6

    Text {
        id: labelItem
        color: ApplicationWindow.window.mutedTextColor
        font.pixelSize: 12
        font.weight: Font.Medium
    }
    TextField {
        id: field
        Accessible.name: root.label
        onEditingFinished: root.editingFinished()
        width: parent.width
        height: 44
        implicitWidth: 120
        color: ApplicationWindow.window.textColor
        placeholderTextColor: ApplicationWindow.window.faintTextColor
        selectByMouse: true
        background: Rectangle {
            radius: 9
            color: ApplicationWindow.window.inputColor
            border.width: field.activeFocus ? 2 : 1
            border.color: field.activeFocus ? ApplicationWindow.window.accentColor : ApplicationWindow.window.borderColor
        }
    }
}
