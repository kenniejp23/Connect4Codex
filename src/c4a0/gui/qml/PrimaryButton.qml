import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Button {
    id: control
    implicitWidth: Math.max(100, contentItem.implicitWidth + 32)
    implicitHeight: 48
    Layout.minimumWidth: implicitWidth
    padding: 12
    leftPadding: 16
    rightPadding: 16
    contentItem: Text {
        text: control.text
        font: control.font
        color: control.enabled ? "#182033" : ApplicationWindow.window.mutedTextColor
        horizontalAlignment: Text.AlignHCenter
        verticalAlignment: Text.AlignVCenter
        elide: Text.ElideRight
    }
    background: Rectangle {
        radius: 8
        color: !control.enabled ? ApplicationWindow.window.panelRaisedColor
             : control.down ? "#D8A72F" : control.hovered ? "#FFDA78" : ApplicationWindow.window.accentColor
        border.width: control.activeFocus ? 2 : 0
        border.color: ApplicationWindow.window.textColor
    }
}
