import logging
import time
from functools import partial
from typing import Tuple

import pymem
from pymem import Pymem

from .env_config import (GAME_NAME, MAX_AGENT_EP, MAX_AGENT_HP, MAX_BOSS_EP,
                         MAX_BOSS_HP, MIN_CODE_LEN, MIN_HELPER_LEN,
                         REVIVE_DELAY)


class CodeInjection():
    def __init__(self, pm: Pymem,
                 original_addr: int, original_code_len: int,
                 helper_addr: int, helper_code_len: int,
                 code_addr: int, injected_code: bytes) -> None:

        self.pm = pm

        self.original_addr = original_addr
        self.original_code = self.pm.read_bytes(original_addr,
                                                original_code_len)

        self.helper_addr = helper_addr
        self.helper_code = self.pm.read_bytes(helper_addr,
                                              helper_code_len)

        if original_code_len < MIN_CODE_LEN \
                or helper_code_len < MIN_HELPER_LEN:
            logging.critical("insufficient code region")
            raise RuntimeError()

        """[helper code]
        NOTE: available bytes >= MIN_HELPER_LEN, 
            better choose "0xcc 0xcc 0xcc ...." region
        
        helper_addr:
            push    rbx
            mov     rbx, code_addr
            jump    rbx
            nop
            ...
        """
        modified_code = b"\x53" + \
            b"\x48\xbb" + code_addr.to_bytes(8, "little") + \
            b"\xff\xe3" + b"\x90" * (helper_code_len - MIN_HELPER_LEN)
        self.pm.write_bytes(helper_addr, modified_code, helper_code_len)

        """[injected code]
        code_addr:
            pop     rbx
            [injected_code]
            [original_code]
            push    rbx
            mov     rbx, original_addr + original_code_len
            jmp     rbx
        """
        injected_code = b"\x5b" + \
            injected_code + self.original_code + b"\x53" + \
            b"\x48\xbb" + (original_addr + 5).to_bytes(8, "little") + \
            b"\xff\xe3"
        self.pm.write_bytes(code_addr, injected_code, len(injected_code))

        """[modified original code]
        NOTE: available bytes >= MIN_CODE_LEN
        
        original_addr:
            jmp     helper_addr
            pop     rbx
            nop
            ...
        """
        modified_code = \
            b"\xe9" + (helper_addr -
                       original_addr - 5).to_bytes(4, "little", signed=True) + \
            b"\x5b" + b"\x90" * (original_code_len - MIN_CODE_LEN)
        self.pm.write_bytes(original_addr, modified_code, original_code_len)

    def __del__(self):
        self.restoreMemory()

    def restoreMemory(self):
        self.pm.write_bytes(self.original_addr,
                            self.original_code, len(self.original_code))
        self.pm.write_bytes(self.helper_addr,
                            self.helper_code, len(self.helper_code))


class Memory():
    def __init__(self) -> None:
        # HACK: Sekiro v1.06
        # NOTE: credit to https://fearlessrevolution.com/viewtopic.php?t=8938
        self.pm = Pymem(f"{GAME_NAME}.exe")
        module_game = pymem.process.module_from_name(
            self.pm.process_handle, f"{GAME_NAME}.exe")

        # NOTE: agent attributes
        """
        E8 ** ** ** ** 48 8B CB
        66 ** ** ** 0F ** ** E8
        ** ** ** ** 66 ** ** **
        0F ** ** F3 ** ** ** 0F
        """
        bytes_pattern = b"\xe8....\x48\x8b\xcb\x66...\x0f..\xe8" \
                        b"....\x66...\x0f..\xf3...\x0f"
        # HACK: sekiro.exe + 0x66888b
        # self.health_read_addr = module_game.lpBaseOfDll + 0x66888b
        health_read_addr = pymem.pattern.pattern_scan_module(
            self.pm.process_handle, module_game, bytes_pattern)
        if health_read_addr is None:
            logging.critical("health_read_addr scan failed")
            raise RuntimeError()
        """[code injection]
        push    rbx
        mov     rbx, agent_mem_addr
        mov     [rbx], rcx
        pop     rbx
        """
        code_addr = self.pm.allocate(256)
        self.agent_mem_ptr = self.pm.allocate(8)  # 8 bytes
        injected_code = b"\x53" + \
            b"\x48\xbb" + self.agent_mem_ptr.to_bytes(8, "little") + \
            b"\x48\x89\x0b" + b"\x5b"
        self.agent_code_injection = CodeInjection(
            self.pm, original_addr=health_read_addr + 5, original_code_len=7,
            helper_addr=health_read_addr + 0xd46, helper_code_len=13,
            code_addr=code_addr, injected_code=injected_code)

        # NOTE: gesture damage
        """
        45 ** ** 89 ** ** ** 00 00 85 DB
        """
        bytes_pattern = b"\x45..\x89...\x00\x00\x85\xdb"
        guard_write_addr = pymem.pattern.pattern_scan_module(
            self.pm.process_handle, module_game, bytes_pattern)
        if guard_write_addr is None:
            logging.critical("guard_write_addr scan failed")
            raise RuntimeError()
        """[code injection]
        push    rbx
        mov     rbx, agent_mem_ptr
        cmp     [rbx], rdi
        je      done
        mov     rbx, boss_mem_ptr
        mov     [rbx], rdi

        done:
        pop     rbx
        """
        code_addr = self.pm.allocate(256)
        self.boss_mem_ptr = self.pm.allocate(8)  # 8 bytes
        injected_code = b"\x53" + \
            b"\x48\xbb" + self.agent_mem_ptr.to_bytes(8, "little") + \
            b"\x48\x39\x3b" + b"\x0f\x84\x0d\x00\x00\x00" + \
            b"\x48\xbb" + self.boss_mem_ptr.to_bytes(8, "little") + \
            b"\x48\x89\x3b" + b"\x5b"
        self.boss_code_injection = CodeInjection(
            self.pm, original_addr=guard_write_addr + 3, original_code_len=6,
            helper_addr=guard_write_addr + 0x33b, helper_code_len=13,
            code_addr=code_addr, injected_code=injected_code)

        # NOTE: live longer
        """
        48 ** ** ** 8B ** 89 ** ** ** 00 00 85 C0 7F
        """
        bytes_pattern = b"\x48...\x8b.\x89...\x00\x00\x85\xc0\x7f"
        health_write_addr = pymem.pattern.pattern_scan_module(
            self.pm.process_handle, module_game, bytes_pattern)
        if health_write_addr is None:
            logging.critical("can't live any longer")
            raise RuntimeError()
        """[code injection]
        push rcx
        push rbx
        push rdx
        mov rdx,dDamageMultiplier
        mov rcx,pPlayer
        cmp [rcx],rbx
        jne @f
        // mov rdx,ddamagemultiplierdefault2
        lea rdx,[rdx+4]
        @@:
        test eax,eax
        jz @f
        lea rbx,[rbx+130]
        cmp [rbx],eax
        jle @f
        mov ecx,[rbx]
        sub ecx,eax
        push rcx
        fild dword ptr [rsp]
        fmul dword ptr [rdx]
        fistp dword ptr [rsp]
        pop rcx
        mov eax,[rbx]
        sub eax,ecx
        jns @f
        xor eax,eax

        @@:
        pop rdx
        pop rbx
        pop rcx
        """
        code_addr = self.pm.allocate(256)
        one = self.pm.allocate(8)  # 8 bytes
        dot_one = self.pm.allocate(8)
        self.pm.write_float(one, 1.0)
        self.pm.write_float(dot_one, 0.1)
        injected_code = b"\x51\x53\x52\x48\xBA" + one.to_bytes(8, "little") + \
            b"\x48\xB9" + self.agent_mem_ptr.to_bytes(8, "little") + b"\x48\x39\x19\x0F\x85\x0D\x00\x00\x00" + \
            b"\x48\xBA" + dot_one.to_bytes(8, "little") + b"\x48\x8D\x12\x85\xC0\x0F\x84\x29\x00\x00" + \
            b"\x00\x48\x8D\x9B\x30\x01\x00\x00\x39\x03\x0F\x8E\x1A\x00" + \
            b"\x00\x00\x8B\x0B\x29\xC1\x51\xDB\x04\x24\xD8\x0A\xDB\x1C" + \
            b"\x24\x59\x8B\x03\x29\xC8\x0F\x89\x02\x00\x00\x00\x31\xC0\x5A\x5B\x59"
        self.health_code_injection = CodeInjection(
            self.pm, original_addr=health_write_addr + 6, original_code_len=6,
            helper_addr=health_write_addr + 0x89B, helper_code_len=13,
            code_addr=code_addr, injected_code=injected_code)

        self.agent_mem_ptr = partial(
            self.pm.read_ulonglong, self.agent_mem_ptr)
        self.boss_mem_ptr = partial(
            self.pm.read_ulonglong, self.boss_mem_ptr)

        # NOTE: automatic boss lock
        # HACK: sekiro.exe + 0x3d78058
        self.state_mem_ptr = partial(
            self.pm.read_ulonglong, module_game.lpBaseOfDll + 0x3d77fb8)
        time.sleep(0.5)

    def restoreMemory(self) -> None:
        self.health_code_injection.restoreMemory()
        self.agent_code_injection.restoreMemory()
        self.boss_code_injection.restoreMemory()

    def resetEndurance(self) -> None:
        try:
            boss_mem_addr = self.boss_mem_ptr()
            self.pm.write_int(boss_mem_addr + 0x148, MAX_BOSS_EP)
        except Exception as e:
            logging.critical(e)
            self.restoreMemory()
            raise RuntimeError()

    def lockBoss(self) -> bool:
        try:
            state_mem_addr = self.state_mem_ptr()
            self.pm.write_bool(state_mem_addr + 0x2831, True)
            time.sleep(0.01)
            lock_state = self.pm.read_bool(state_mem_addr + 0x2831)
            return lock_state
        except Exception as e:
            logging.critical(e)
            self.restoreMemory()
            raise RuntimeError()

    def transportAgent(self, loc: Tuple[float, float, float]):
        try:
            def nextPtr(x):
                return self.pm.read_ulonglong(x)
            coord_mem_addr = self.agent_mem_ptr() + 0x8
            offsets = [0x1ff8, 0x68, 0x80]
            for offset in offsets:
                coord_mem_addr = nextPtr(coord_mem_addr) + offset
            for i in range(len(loc)):
                self.pm.write_float(coord_mem_addr + 0x4 * i, loc[i])
        except Exception as e:
            logging.critical(e)
            self.restoreMemory()
            raise RuntimeError()

    def getStatus(self) -> Tuple[float, float, float]:
        """[summary]

        Returns:
            agent hp    [0, 1]
            agent ep    [0, 1]
            boss hp     [0, 1]
        """
        try:
            agent_mem_addr = self.agent_mem_ptr()
            agent_hp = self.pm.read_int(agent_mem_addr + 0x130)
            agent_ep = self.pm.read_int(agent_mem_addr + 0x148)
            boss_mem_addr = self.boss_mem_ptr()
            boss_hp = self.pm.read_int(boss_mem_addr + 0x130)
            return (agent_hp / MAX_AGENT_HP, agent_ep / MAX_AGENT_EP, boss_hp / MAX_BOSS_HP)
        except Exception as e:
            logging.critical(e)
            self.restoreMemory()
            raise RuntimeError()

    def reviveAgent(self, need_delay: bool) -> None:
        try:
            agent_mem_addr = self.agent_mem_ptr()
            self.pm.write_int(agent_mem_addr + 0x130, MAX_AGENT_HP)
            time.sleep(need_delay * REVIVE_DELAY)
        except Exception as e:
            logging.critical(e)
            self.restoreMemory()
            raise RuntimeError()

    def reviveBoss(self) -> None:
        try:
            boss_mem_addr = self.boss_mem_ptr()
            self.pm.write_int(boss_mem_addr + 0x130, MAX_BOSS_HP)
        except Exception as e:
            logging.critical(e)
            self.restoreMemory()
            raise RuntimeError()
